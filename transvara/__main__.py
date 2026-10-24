from __future__ import annotations

import gc
import os
import random
import sys
import warnings
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RISE" / "src"))

from rise.nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from rise.nn.models import ContextualSvaraClassifier

from .genetic import transcribe
from .metrics import (
    in_coverage,
    note_metrics,
    notes_to_events,
    phrase_rows,
    reference_events,
)
from .plots import plot_transcription, plot_windows
from .preprocess import (
    beat_spacing,
    build_frames,
    coverage_frames,
    frame_spans_to_seconds,
    make_encoder,
    merge_coverage,
    read_annotations,
    sample_none_spans,
    tonic_for,
)

console = Console(stderr=True)
warnings.filterwarnings("ignore", message=".*notes are empty.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

TRAIN_PERFORMER = "kamakshi"
TEST_PERFORMER = "raksha_bettare"
RISE_RUN = "classification_bhairavi"

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SVARAS: tuple[str, ...] = tuple(CONFIG["svaras"])
EMBED_DIM = CONFIG["embed_dim"]
EMBED_DEPTH = CONFIG["embed_depth"]
CONTEXT_DIM = EMBED_DIM * 3
CONTEXT_DURATION = CONFIG["context_duration"]
ENCODE_BATCH = CONFIG["encode_batch"]
MIN_FRAMES = CONFIG["min_frames"]
ALPHA = CONFIG["alpha"]
GAMMA = CONFIG["gamma"]


def main() -> None:
    args = parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    console.print("[bold]TRANSVARA[/bold]")
    table = Table()
    table.add_column("option")
    table.add_column("value")
    for key, value in {
        "cost": "L = (alpha * Lsvara + beta * Ltrans) / (alpha + beta) + gamma * Llength",
        "alpha / beta / gamma": f"{ALPHA} / {1.0 - ALPHA} / {GAMMA}",
        "pm / pc / pd / pa": f"{args.pm} / {args.pc} / {args.pd} / {args.pa}",
        "pj / join threshold": f"{args.pj} / {args.join_threshold}",
        "ga population": args.ga_population,
        "ga iterations": args.ga_iterations,
        "dmin / dmax / smax": "16th beat / whole note / one beat (TCN)",
        "model": "RISE fine-tuned contextual classifier (0.5 s melodic context)",
        "match window": "half beat (paper)",
        "sample rate": args.sample_rate,
        "seed": args.seed,
        "device": device,
    }.items():
        table.add_row(key, str(value))
    console.print(table)

    console.print("\n[bold]Data[/bold]")
    progress = Progress(console=console)
    task = progress.add_task("data", total=2)
    progress.start()
    progress.refresh()
    performances = []
    for performer in (TRAIN_PERFORMER, TEST_PERFORMER):
        progress.update(task, description=performer)
        performances.append(
            {
                "performer": performer,
                **build_frames(
                    DATA_DIR / "pitch_tracks" / f"{performer}.tsv",
                    tonic_for(DATA_DIR / "tonics.yaml", performer),
                ),
            }
        )
        progress.advance(task)
    progress.stop()
    console.print(f"train: {TRAIN_PERFORMER} test: {TEST_PERFORMER}")

    beat_path = DATA_DIR / "beat_tracks" / f"{TEST_PERFORMER}.tsv"
    if not beat_path.exists():
        raise SystemExit(f"no beat track at {beat_path}")
    beat_frame = pd.read_csv(beat_path, sep="\t", header=None, names=["time", "beat"])
    held = next(item for item in performances if item["performer"] == TEST_PERFORMER)
    held["beats"] = torch.from_numpy(beat_frame["time"].to_numpy(dtype=np.float64))
    held["downbeats"] = torch.from_numpy(
        beat_frame.loc[beat_frame["beat"] == 1, "time"].to_numpy(dtype=np.float64)
    )

    training = [item for item in performances if item["performer"] == TRAIN_PERFORMER]
    progress = Progress(console=console)
    task = progress.add_task("pipeline", total=3)
    progress.start()
    progress.refresh()

    progress.update(task, description="classifier")
    console.print("\n[bold]model[/bold]: loading fine-tuned RISE model")
    model = ContextualSvaraClassifier(EMBED_DIM, EMBED_DEPTH, len(SVARAS))
    model.apply_lora(DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
    state = torch.load(
        ROOT / "RISE" / ".cache" / "runs" / RISE_RUN / "best.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    del state
    model.eval()
    model.to(device)
    encode = make_encoder(model, device, MIN_FRAMES, CONTEXT_DURATION, ENCODE_BATCH)

    features: list = []
    labels: list = []
    combined_features: list = []
    counts, bigrams, unigrams = Counter(), Counter(), Counter()
    for item in training:
        times = item["times"].numpy()
        contour = item["contour"].numpy()
        begins, finishes, annotations = read_annotations(
            DATA_DIR / "annotations" / f"{item['performer']}.tsv"
        )
        sequence, spans = [], []
        for begin, finish, annotation in zip(
            begins, finishes, annotations, strict=True
        ):
            if annotation[0] not in SVARAS:
                continue
            if (
                int(np.searchsorted(times, finish))
                - int(np.searchsorted(times, begin))
                < 2
            ):
                continue
            spans.append((float(begin), float(finish)))
            index = SVARAS.index(annotation[0])
            labels.append(index)
            sequence.append(index)
        features.extend(encode(times, contour, spans))
        combined = [
            (group[0][0], group[-1][1])
            for size in (2, 3)
            for group in zip(*(spans[i:] for i in range(size)))
            if group[-1][1] > group[0][0]
            and all(
                group[i + 1][0] - group[i][1] <= CONTEXT_DURATION
                for i in range(size - 1)
            )
        ]
        if combined:
            combined_features.extend(encode(times, contour, combined))
        counts.update(zip(sequence, sequence[1:], sequence[2:]))
        bigrams.update(zip(sequence[:-2], sequence[1:-1]))
        unigrams.update(sequence)
    annotated = len(features)
    rng.shuffle(combined_features)
    quota = max(annotated // len(SVARAS), 1)
    none_features: list = combined_features[: max(quota // 2, 1)]
    attempt = 0
    while len(none_features) < quota:
        item = training[attempt % len(training)]
        attempt += 1
        if attempt > 20 * len(training) and not none_features:
            raise SystemExit("no annotated coverage found in training performances")
        times = item["times"].numpy()
        contour = item["contour"].numpy()
        begins, finishes, _ = read_annotations(
            DATA_DIR / "annotations" / f"{item['performer']}.tsv"
        )
        spans = coverage_frames(merge_coverage(begins, finishes), times)
        if not spans:
            continue
        sampled = sample_none_spans(spans, max(quota // len(training), 1), rng)
        none_features.extend(
            encode(times, contour, frame_spans_to_seconds(sampled, times))
        )
    none_features = none_features[:quota]
    features.extend(none_features)
    labels.extend([len(SVARAS)] * len(none_features))
    base = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    classifier = make_pipeline(
        StandardScaler(), CalibratedClassifierCV(base, method="isotonic", cv=3)
    )
    classifier.fit(
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )

    classes = len(SVARAS) + 1
    trigram = np.zeros((classes, classes, classes))
    bigram = np.zeros((classes, classes))
    unigram = np.zeros(classes)
    for array, counter in (
        (trigram, counts),
        (bigram, bigrams),
        (unigram, unigrams),
    ):
        for key, value in counter.items():
            array[key] = value
    unigram = (unigram + 1e-3) / (unigram.sum() + 1e-3 * classes)
    console.print(
        f"{TRAIN_PERFORMER}: {annotated} annotated + {len(none_features)} none segments"
    )
    progress.advance(task)

    times = held["times"].numpy()
    begins, finishes, annotations = read_annotations(
        DATA_DIR / "annotations" / f"{TEST_PERFORMER}.tsv"
    )
    coverage = merge_coverage(begins, finishes)
    regions = coverage_frames(coverage, times)
    beats, beat_frames, _ = beat_spacing(held["beats"], held["times"])
    progress.update(task, description="transcribe")
    console.print(f"{TEST_PERFORMER}: transcribing {len(regions)} annotated regions")
    held["notes"] = transcribe(
        encode,
        classifier,
        trigram,
        bigram,
        unigram,
        args,
        rng,
        held,
        regions,
        beats,
        beat_frames,
        [ALPHA, 1.0 - ALPHA, GAMMA],
        SVARAS,
    )
    progress.advance(task)

    held["reference"] = reference_events(times, begins, finishes, annotations)
    held["coverage"] = coverage
    phrases = merge_coverage(begins, finishes, 0.5)
    window = (
        float(np.median(np.diff(held["beats"].numpy()))) / 2.0
        if len(held["beats"]) > 1
        else 0.1
    )
    held["estimated"] = in_coverage(
        notes_to_events(times, held["notes"]), coverage
    )
    held["metrics"] = note_metrics(held["reference"], held["estimated"], window)
    progress.update(task, description="plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    windows = plot_windows(phrases)
    for index, (begin, finish) in enumerate(windows, start=1):
        transcription_path = PLOTS_DIR / f"{TEST_PERFORMER}_region{index:03d}.png"
        plot_transcription(
            held["times"].numpy(),
            held["contour"].numpy(),
            [
                ("ground truth", held["reference"]),
                ("genetic algorithm", held["estimated"]),
            ],
            (begin, finish),
            f"transcription ({TEST_PERFORMER}, region {index}/{len(windows)} · {begin:.1f}-{finish:.1f}s)",
            transcription_path,
            coverage,
            held["beats"].numpy(),
            held["downbeats"].numpy(),
        )
        console.print(f"figure written to {transcription_path}")
    progress.advance(task)

    del model, encode
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    progress.stop()

    console.print("\n[bold]Evaluation[/bold]")
    rows = phrase_rows(phrases, held["reference"], held["estimated"], window)
    table = Table(title=f"Top {len(rows)} longest test phrases")
    for column in ("Transcription", "Annotation", "Lev Norm", "LCS", "LCS Norm", "AOR"):
        table.add_column(column)
    for transcription, annotation, lev, common, lcs, aor in rows:
        table.add_row(
            transcription,
            annotation,
            f"{lev:.2f}",
            common,
            f"{lcs:.2f}",
            f"{aor:.2f}",
        )
    console.print(table)

    table = Table(title="Metrics")
    table.add_column("Metric")
    for label in ("COn", "COnP", "COnOff", "COnPOff"):
        table.add_column(label, justify="right")
    for label, suffix in (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")):
        table.add_row(
            label,
            *(
                f"{held['metrics'][f'{key}_{suffix}']:.3f}"
                for key in ("con", "conp", "conoff", "conpoff")
            ),
        )
    console.print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    phrases_path = RESULTS_DIR / "phrases.csv"
    pd.DataFrame(rows, columns=["transcription", "annotation", "lev_norm", "lcs", "lcs_norm", "aor"]).to_csv(
        phrases_path, index=False
    )
    console.print(f"table written to {phrases_path}")

    metrics_path = RESULTS_DIR / "metrics.csv"
    pd.DataFrame(
        [
            (
                label,
                *(
                    held["metrics"][f"{key}_{suffix}"]
                    for key in ("con", "conp", "conoff", "conpoff")
                ),
            )
            for label, suffix in (
                ("F1", "f1"),
                ("Precision", "precision"),
                ("Recall", "recall"),
            )
        ],
        columns=["metric", "COn", "COnP", "COnOff", "COnPOff"],
    ).to_csv(metrics_path, index=False)
    console.print(f"table written to {metrics_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TRANSVARA")
    parser.add_argument("--pm", type=float, default=0.75)
    parser.add_argument("--pc", type=float, default=0.75)
    parser.add_argument("--pd", type=float, default=0.5)
    parser.add_argument("--pa", type=float, default=0.75)
    parser.add_argument("--pj", type=float, default=0.5)
    parser.add_argument("--join-threshold", type=float, default=0.6)
    parser.add_argument("--ga-population", type=int, default=40)
    parser.add_argument("--ga-iterations", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

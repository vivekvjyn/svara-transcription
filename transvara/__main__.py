from __future__ import annotations

import gc
import json
import os
import random
import sys
import warnings
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

import compiam
import joblib
import Levenshtein
import librosa
import mir_eval
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

from rise.data.corpora import RAGA_SVARAS
from rise.nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from rise.nn.models import ContextualSvaraClassifier

from .genetic import GeneticOptimiser
from .plots import plot_region, plot_transcription
from .preprocess import (
    build_frames,
    coverage_frames,
    frame_spans_to_seconds,
    make_encoder,
    merge_coverage,
    read_annotations,
    sample_none_spans,
)

console = Console(stderr=True)
warnings.filterwarnings("ignore", message=".*notes are empty.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

RAGAS: tuple[str, ...] = (
    "abhogi",
    "begada",
    "kalyani",
    "mohanam",
    "sahana",
    "saveri",
    "sri",
)

TEST_PERFORMER: dict[str, str] = {
    "abhogi": "vignesh",
    "begada": "prasanna",
    "kalyani": "ramakrishnamurthy",
    "mohanam": "dharini",
    "sahana": "sreevidya",
    "saveri": "badrinarayanan",
    "sri": "vignesh",
}

VARIANTS = (
    "Lsvara",
    "Lsvara+Ltrans",
    "Lsvara+Ltrans+Llength",
    "Lsvara+Ltrans+Llength+Lnone",
)
MEAN_KEYS = (
    "con_f1",
    "con_precision",
    "con_recall",
    "conp_f1",
    "conoff_f1",
    "conpoff_f1",
    "aor",
    "levenshtein_norm",
    "lcs_norm",
)
MEAN_LABELS = (
    "COn F",
    "COn P",
    "COn R",
    "COnP F",
    "COnOff F",
    "COnPOff F",
    "AOR",
    "Lev norm",
    "LCS norm",
)
ROW_KEYS = (
    "con_f1",
    "conp_f1",
    "conoff_f1",
    "conpoff_f1",
    "aor",
    "levenshtein_norm",
    "lcs_norm",
)
ROW_LABELS = ("COn F", "COnP F", "COnOff F", "COnPOff F", "AOR", "Lev norm", "LCS norm")

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SVARAS: tuple[str, ...] = tuple(CONFIG["svaras"])
PLOT_SECONDS = CONFIG["plot_seconds"]
EMBED_DIM = CONFIG["embed_dim"]
EMBED_DEPTH = CONFIG["embed_depth"]
CONTEXT_DIM = EMBED_DIM * 3
CONTEXT_DURATION = CONFIG["context_duration"]
ENCODE_BATCH = CONFIG["encode_batch"]
MIN_FRAMES = CONFIG["min_frames"]
ALPHA = CONFIG["alpha"]
GAMMA = CONFIG["gamma"]
DELTA = CONFIG["delta"]


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
        "cost": "L = (alpha * Lsvara + beta * Ltrans) / (alpha + beta) + gamma * Llength - delta * Lnone",
        "alpha / beta / gamma / delta": f"{ALPHA} / {1.0 - ALPHA} / {GAMMA} / {DELTA}",
        "ablation": "Lsvara / Lsvara+Ltrans / Lsvara+Ltrans+Llength / Lsvara+Ltrans+Llength+Lnone",
        "pm / pc / pd / pa": f"{args.pm} / {args.pc} / {args.pd} / {args.pa}",
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
    tonics = yaml.safe_load((DATA_DIR / "tonics.yaml").read_text())
    progress = Progress(console=console)
    task = progress.add_task(
        "data",
        total=sum(
            len(list((DATA_DIR / "pitch_tracks" / raga).glob("*.tsv")))
            for raga in RAGAS
        ),
    )
    progress.start()
    performances = []
    for raga in RAGAS:
        for pitch_path in sorted((DATA_DIR / "pitch_tracks" / raga).glob("*.tsv")):
            performer = pitch_path.stem
            progress.update(task, description=f"{raga}/{performer}")
            frames_cache = CACHE_DIR / "datasets" / "frames" / raga / f"{performer}.pt"
            if frames_cache.exists():
                stored = torch.load(frames_cache, map_location="cpu", weights_only=False)
            else:
                stored = build_frames(DATA_DIR, raga, performer, tonics)
                frames_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stored, frames_cache)
            performances.append(
                {
                    "raga": raga,
                    "performer": performer,
                    "times": stored["times"],
                    "contour": stored["contour"].to(torch.float32),
                }
            )
            progress.advance(task)
    progress.stop()
    console.print(f"{len(performances)} performances")
    available = {
        raga: {p["performer"] for p in performances if p["raga"] == raga}
        for raga in RAGAS
    }
    for raga in RAGAS:
        if TEST_PERFORMER[raga] not in available[raga]:
            raise SystemExit(
                f"TEST_PERFORMER[{raga}] = {TEST_PERFORMER[raga]} not in "
                f"{sorted(available[raga])}"
            )
    console.print(
        "test: " + ", ".join(f"{raga}/{TEST_PERFORMER[raga]}" for raga in RAGAS)
    )

    console.print("\n[bold]Beat tracking[/bold]")
    beat_tracker = compiam.load_model(
        "rhythm:tcn-carnatic",
        gpu="0" if device.type == "cuda" else "-1",
        # the joint post-processor decodes four bar-length DBNs whose Viterbi
        # density matrix is frames x states float64 (~10 GB on 8-minute audio);
        # we only keep beat times, so the sequential path is equivalent for us
        # and decodes a single ~5.7k-state DBN (~3 GB peak)
        post_processor="sequential",
    )
    console.print(
        f"Loaded compIAM tcn_carnatic beat tracker from {beat_tracker.model_path}"
    )
    progress = Progress(console=console)
    task = progress.add_task("beats", total=len(performances))
    progress.start()
    for performance in performances:
        raga, performer = performance["raga"], performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        beats_cache = CACHE_DIR / "beats" / raga / f"{performer}.pt"
        if beats_cache.exists():
            beats = torch.load(beats_cache, map_location="cpu", weights_only=False)[
                "beats"
            ]
        else:
            source = DATA_DIR / "audio" / raga / f"{performer}.mp3"
            signal, _ = librosa.load(source, sr=args.sample_rate, mono=True)
            prediction = beat_tracker.predict(signal, args.sample_rate).reshape(-1, 2)
            beats = torch.from_numpy(np.ascontiguousarray(prediction[:, 0]))
            beats_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"beats": beats}, beats_cache)
            del signal, prediction
        performance["beats"] = beats
        progress.advance(task)
    progress.stop()
    del beat_tracker
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    classifier_dir = CACHE_DIR / "classifier"
    markov_dir = CACHE_DIR / "markov"
    genetic_dir = CACHE_DIR / "genetic"
    test_performances = [
        item
        for item in performances
        if item["performer"] == TEST_PERFORMER[item["raga"]]
    ]
    progress = Progress(console=console)
    task = progress.add_task("pipeline", total=len(RAGAS) * 6)
    progress.start()
    for raga in RAGAS:
        test = TEST_PERFORMER[raga]
        training = [
            item
            for item in performances
            if item["raga"] == raga and item["performer"] != test
        ]

        progress.update(task, description=f"{raga}: classifier")
        console.print(f"\n[bold]{raga}[/bold]: loading fine-tuned RISE model")
        model = ContextualSvaraClassifier(
            EMBED_DIM, EMBED_DEPTH, len(RAGA_SVARAS[raga])
        )
        model.apply_lora(DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
        state = torch.load(
            ROOT
            / "RISE"
            / ".cache"
            / "runs"
            / f"classification_{raga}_pretrained"
            / "best.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        del state
        model.eval()
        model.to(device)
        encode = make_encoder(model, device, MIN_FRAMES, CONTEXT_DURATION, ENCODE_BATCH)

        classifier_path = classifier_dir / f"{raga}.joblib"
        markov_path = markov_dir / f"{raga}.npz"
        cached = False
        classifier = trigram = bigram = unigram = None
        if classifier_path.exists() and markov_path.exists():
            candidate = joblib.load(classifier_path)
            if candidate.named_steps["standardscaler"].n_features_in_ == CONTEXT_DIM:
                classifier = candidate
                stored = np.load(markov_path)
                trigram = stored["trigram"]
                bigram = stored["bigram"]
                unigram = stored["unigram"]
                cached = True
                console.print(f"{raga}: cached classifier and Markov model")
        if not cached:
            features: list = []
            labels: list = []
            combined_features: list = []
            counts, bigrams, unigrams = Counter(), Counter(), Counter()
            for item in training:
                times = item["times"].numpy()
                contour = item["contour"].numpy()
                begins, finishes, annotations = read_annotations(
                    DATA_DIR / "annotations" / raga / f"{item['performer']}_{raga}.tsv"
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
                    raise SystemExit(
                        f"{raga}: no annotated coverage found in training performances"
                    )
                times = item["times"].numpy()
                contour = item["contour"].numpy()
                begins, finishes, _ = read_annotations(
                    DATA_DIR / "annotations" / raga / f"{item['performer']}_{raga}.tsv"
                )
                spans = coverage_frames(merge_coverage(begins, finishes), times)
                if not spans:
                    continue
                sampled = sample_none_spans(
                    spans, max(quota // len(training), 1), rng
                )
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
            classifier_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(classifier, classifier_path)

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
            markov_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(markov_path, trigram=trigram, bigram=bigram, unigram=unigram)
            console.print(
                f"{raga}: {annotated} annotated + {len(none_features)} none segments, "
                f"trained on {len(training)} performances (test {test})"
            )
        progress.advance(task)

        transcribe = transcriber(encode, classifier, trigram, bigram, unigram, args, rng)

        held = next(item for item in test_performances if item["raga"] == raga)
        times = held["times"].numpy()
        begins, finishes, annotations = read_annotations(
            DATA_DIR / "annotations" / raga / f"{test}_{raga}.tsv"
        )
        coverage = merge_coverage(begins, finishes)
        regions = coverage_frames(coverage, times)
        beats, beat_frames, _ = beat_spacing(held["beats"], held["times"])
        held["notes"] = {}
        for variant, cost in (
            ("Lsvara", [1.0, 0.0, 0.0, 0.0]),
            ("Lsvara+Ltrans", [ALPHA, 1.0 - ALPHA, 0.0, 0.0]),
            ("Lsvara+Ltrans+Llength", [ALPHA, 1.0 - ALPHA, GAMMA, 0.0]),
            (
                "Lsvara+Ltrans+Llength+Lnone",
                [ALPHA, 1.0 - ALPHA, GAMMA, DELTA],
            ),
        ):
            progress.update(task, description=f"{raga}: {variant}")
            held["notes"][variant] = transcribe(
                genetic_dir / raga / f"{test}_{variant}.pt",
                held,
                regions,
                beats,
                beat_frames,
                cost,
                f"{raga}/{test} [{variant}]",
            )
            progress.advance(task)

        held["reference"] = reference_events(times, begins, finishes, annotations)
        held["coverage"] = coverage
        window = (
            float(np.median(np.diff(held["beats"].numpy()))) / 2.0
            if len(held["beats"]) > 1
            else 0.1
        )
        progress.update(task, description=f"{raga}: plots")
        region = plot_region(coverage, times, PLOT_SECONDS)
        signal, _ = librosa.load(
            DATA_DIR / "audio" / raga / f"{test}.mp3",
            sr=args.sample_rate,
            mono=True,
        )
        for variant in VARIANTS:
            estimated = in_coverage(
                notes_to_events(times, held["notes"][variant]), coverage
            )
            held.setdefault("estimated", {})[variant] = estimated
            held.setdefault("metrics", {})[variant] = note_metrics(
                held["reference"], estimated, window
            )
            transcription_path = PLOTS_DIR / raga / f"{test}_{variant}_transcription.png"
            transcription_path.parent.mkdir(parents=True, exist_ok=True)
            plot_transcription(
                signal,
                args.sample_rate,
                [
                    ("ground truth", held["reference"]),
                    ("genetic algorithm", estimated),
                ],
                region,
                f"transcription ({test}, {variant})",
                transcription_path,
                coverage,
            )
            console.print(f"figure written to {transcription_path}")
        del signal
        progress.advance(task)

        del model, encode
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.stop()

    console.print("\n[bold]Evaluation[/bold]")
    progress = Progress(console=console)
    task = progress.add_task(
        "evaluation", total=len(VARIANTS) * len(test_performances)
    )
    progress.start()
    rows_by_variant: dict[str, list[tuple]] = {}
    means_by_variant: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        rows: list[tuple] = []
        aggregated: dict[str, dict[str, list[float]]] = {}
        for performance in test_performances:
            raga, performer = performance["raga"], performance["performer"]
            progress.update(task, description=f"{variant}: {raga}/{performer}")
            metrics = performance["metrics"][variant]
            for key, value in metrics.items():
                aggregated.setdefault(raga, {}).setdefault(key, []).append(value)
            rows.append(
                (raga, performer, *(f"{metrics[key]:.3f}" for key in ROW_KEYS))
            )
            progress.advance(task)
        rows_by_variant[variant] = rows
        means = {
            raga: {key: float(np.mean(values)) for key, values in aggregated[raga].items()}
            for raga in RAGAS
        }
        means["all"] = {
            key: float(
                np.mean([value for raga in RAGAS for value in aggregated[raga][key]])
            )
            for key in aggregated[RAGAS[0]]
        }
        means_by_variant[variant] = means

        table = Table(title=f"Mean [{variant}] (paper metrics)")
        table.add_column("raga")
        for label in MEAN_LABELS:
            table.add_column(label, justify="right")
        for raga in [*RAGAS, "all"]:
            table.add_row(raga, *(f"{means[raga][key]:.3f}" for key in MEAN_KEYS))
        console.print(table)
    progress.stop()

    table = Table(title="Ablation (mean over test set)")
    table.add_column("variant")
    for label in ROW_LABELS:
        table.add_column(label, justify="right")
    for variant in VARIANTS:
        values = means_by_variant[variant]["all"]
        table.add_row(variant, *(f"{values[key]:.3f}" for key in ROW_KEYS))
    console.print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        means = means_by_variant[variant]
        results_path = RESULTS_DIR / f"results_{variant}.json"
        results_path.write_text(
            json.dumps(
                {
                    "alpha": ALPHA,
                    "beta": 1.0 - ALPHA,
                    "gamma": GAMMA,
                    "delta": DELTA,
                    "mean": means["all"],
                    "per_raga": {
                        raga: {
                            "mean": means[raga],
                            "per_performance": [
                                {
                                    "performer": item["performer"],
                                    "metrics": item["metrics"][variant],
                                }
                                for item in test_performances
                                if item["raga"] == raga
                            ],
                        }
                        for raga in RAGAS
                    },
                },
                indent=4,
            )
        )
        console.print(f"results written to {results_path}")

        metrics_path = RESULTS_DIR / f"metrics_{variant}.csv"
        pd.DataFrame(
            rows_by_variant[variant],
            columns=["raga", "performer", *ROW_LABELS],
        ).to_csv(metrics_path, index=False)
        console.print(f"table written to {metrics_path}")

        mean_path = RESULTS_DIR / f"mean_{variant}.csv"
        pd.DataFrame(
            [
                (raga, key.replace("_", " "), value)
                for raga in [*RAGAS, "all"]
                for key, value in sorted(means[raga].items())
            ],
            columns=["raga", "metric", "value"],
        ).to_csv(mean_path, index=False)
        console.print(f"table written to {mean_path}")

    ablation_path = RESULTS_DIR / "ablation.csv"
    pd.DataFrame(
        [
            (variant, key.replace("_", " "), means_by_variant[variant]["all"][key])
            for variant in VARIANTS
            for key in ROW_KEYS
        ],
        columns=["variant", "metric", "value"],
    ).to_csv(ablation_path, index=False)
    console.print(f"table written to {ablation_path}")


def notes_to_events(
    times: np.ndarray, notes: list[tuple[int, int, int]]
) -> list[tuple[float, float, str]]:
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    return [
        (float(times[start]), float(times[stop - 1]) + step, SVARAS[label])
        for start, stop, label in notes
    ]


def reference_events(
    times: np.ndarray,
    begins: np.ndarray,
    finishes: np.ndarray,
    annotations: list[str],
) -> list[tuple[float, float, str]]:
    return [
        (float(begins[index]), float(finishes[index]), annotations[index][0])
        for index in range(len(annotations))
        if annotations[index][0] in SVARAS
        and np.any((times >= begins[index]) & (times < finishes[index]))
    ]


def in_coverage(
    events: list[tuple[float, float, str]], coverage: list[tuple[float, float]]
) -> list[tuple[float, float, str]]:
    return [
        event
        for event in events
        if any(event[0] >= begin and event[0] < finish for begin, finish in coverage)
    ]


def lcs_length(first: str, second: str) -> int:
    previous = [0] * (len(second) + 1)
    for character in first:
        current = [0]
        for index, other in enumerate(second):
            current.append(
                previous[index] + 1
                if character == other
                else max(current[-1], previous[index + 1])
            )
        previous = current
    return previous[-1]


def note_metrics(
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    window: float,
) -> dict[str, float]:
    ref_intervals = np.array(
        [[note[0], note[1]] for note in reference], dtype=float
    ).reshape(-1, 2)
    est_intervals = np.array(
        [[note[0], note[1]] for note in estimated], dtype=float
    ).reshape(-1, 2)
    ref_pitches = np.array(
        [440.0 * 2 ** (SVARAS.index(note[2]) / 12) for note in reference]
    )
    est_pitches = np.array(
        [440.0 * 2 ** (SVARAS.index(note[2]) / 12) for note in estimated]
    )
    flat_ref = np.full(len(reference), 440.0)
    flat_est = np.full(len(estimated), 440.0)
    scores = {
        "con": mir_eval.transcription.onset_precision_recall_f1(
            ref_intervals, est_intervals, onset_tolerance=window
        ),
        "conp": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=None,
        ),
        "conoff": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            flat_ref,
            est_intervals,
            flat_est,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=0.0,
            offset_min_tolerance=window,
        ),
        "conpoff": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=0.0,
            offset_min_tolerance=window,
        ),
    }
    metrics: dict[str, float] = {}
    for key, values in scores.items():
        metrics[f"{key}_precision"] = float(values[0])
        metrics[f"{key}_recall"] = float(values[1])
        metrics[f"{key}_f1"] = float(values[2])
    metrics["aor"] = float(scores["conpoff"][3])

    reference_text = "".join(note[2] for note in reference)
    estimated_text = "".join(note[2] for note in estimated)
    distance = Levenshtein.distance(reference_text, estimated_text)
    lcs = lcs_length(reference_text, estimated_text)
    metrics["levenshtein"] = float(distance)
    metrics["levenshtein_norm"] = distance / max(len(reference_text), 1)
    metrics["lcs"] = float(lcs)
    metrics["lcs_norm"] = lcs / max(len(reference_text), 1)
    metrics["num_reference_notes"] = float(len(reference))
    metrics["num_estimated_notes"] = float(len(estimated))
    return metrics


def beat_spacing(
    beats: torch.Tensor, times: torch.Tensor
) -> tuple[np.ndarray, float, float]:
    beat_times = beats.numpy()
    times_array = times.numpy()
    step = float(times_array[1] - times_array[0]) if len(times_array) > 1 else 0.0
    frames = np.clip(
        np.searchsorted(times_array, beat_times), 0, max(len(times_array) - 1, 0)
    )
    if len(beat_times) > 1:
        interval = float(np.median(np.diff(beat_times)))
        return frames, (interval / step if step > 0 else 50.0), interval / 2.0
    return frames, 50.0, 0.1


def transcriber(encode, classifier, trigram, bigram, unigram, args, rng):
    def transcribe(cache_path, performance, regions, beats, beat_frames, cost, label):
        times = performance["times"].numpy()
        contour = performance["contour"].numpy()

        def embed_fn(spans):
            return encode(times, contour, frame_spans_to_seconds(spans, times))

        if cache_path.exists():
            stored = torch.load(cache_path, map_location="cpu", weights_only=False)
            if stored.get("regions") == regions and stored.get("cost") == list(cost):
                console.print(
                    f"{label}: cached transcription ({len(stored['notes'])} notes)"
                )
                return stored["notes"]
        console.print(f"{label}: transcribing {len(regions)} annotated regions")
        optimiser = GeneticOptimiser(
            embed_fn, classifier, trigram, bigram, unigram, SVARAS, args, rng
        )
        notes = optimiser.transcribe(
            contour, regions, beats, beat_frames, cost[0], cost[1], cost[2], cost[3]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"notes": notes, "regions": regions, "cost": list(cost)}, cache_path)
        return notes

    return transcribe


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TRANSVARA")
    parser.add_argument("--pm", type=float, default=0.75)
    parser.add_argument("--pc", type=float, default=0.75)
    parser.add_argument("--pd", type=float, default=0.5)
    parser.add_argument("--pa", type=float, default=0.75)
    parser.add_argument("--ga-population", type=int, default=40)
    parser.add_argument("--ga-iterations", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

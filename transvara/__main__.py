"""TRANSVARA: transcribe the test performance of each raga and print the paper tables.

For each of the seven ragas this prints exactly two markdown tables for the
reported variant (``Lsvara+Ltrans+Llength``): the note metrics and the longest
transcribed phrases. Everything else of the original pipeline - the options
banner, progress bars, plots, ablations, and result files - is gone; the
compiam TCN beat tracker is gone too (beats are read from
``data/beat_tracks/<raga>/<performer>.tsv``), and the RISE submodule has been
replaced by the vendored model code in ``transvara/nn`` with checkpoints in
``model_weights/``.

The heavy resources (encoder, classifier, genetic search) are built only when a
cache is missing, so with warm caches a run costs a few seconds.
"""

from __future__ import annotations

import random
import warnings
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from .genetic import GeneticOptimiser
from .metrics import (
    in_coverage,
    note_metrics,
    notes_to_events,
    phrase_rows,
    reference_events,
)
from .preprocess import (
    RAGA_SVARAS,
    build_frames,
    coverage_frames,
    frame_spans_to_seconds,
    make_encoder,
    merge_coverage,
    read_annotations,
    read_beats,
    sample_none_spans,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
WEIGHTS_DIR = ROOT / "model_weights"
TRACKS_DIR = DATA_DIR / "beat_tracks"

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

#: The single variant reported in the paper.
VARIANT = "Lsvara+Ltrans+Llength"

#: Phrases listed in the second table.
PHRASE_COUNT = 10

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

#: The loss of the reported variant,
#: ``L = (alpha * Lsvara + beta * Ltrans) / (alpha + beta) + gamma * Llength``.
COST: list[float] = [ALPHA, 1.0 - ALPHA, GAMMA, 0.0]


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TRANSVARA")
    parser.add_argument("--pm", type=float, default=0.75)
    parser.add_argument("--pc", type=float, default=0.75)
    parser.add_argument("--pd", type=float, default=0.5)
    parser.add_argument("--pa", type=float, default=0.75)
    parser.add_argument("--ga-population", type=int, default=40)
    parser.add_argument("--ga-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def frames_for(raga: str, performer: str, tonics: dict) -> dict[str, torch.Tensor]:
    """Contours of one performance from the cache, rebuilt from the pitch track on a miss."""
    cache = CACHE_DIR / "datasets" / "frames" / raga / f"{performer}.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        stored = build_frames(DATA_DIR, raga, performer, tonics)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stored, cache)
    stored["contour"] = stored["contour"].to(torch.float32)
    return stored


def beat_spacing(
    beats: torch.Tensor, times: torch.Tensor
) -> tuple[np.ndarray, float, float]:
    """Frame index of each beat, frames per beat, seconds per beat."""
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


def raga_resources(
    raga: str, args: Namespace, rng: random.Random, tonics: dict
) -> tuple:
    """Encoder, svara classifier and transition model for one raga (slow path).

    Built only when a genetic cache is missing: the checkpoint in
    ``model_weights/`` feeds the vendored ``ContextualSvaraClassifier``, whose
    features train the classifier cached under ``.cache/classifier``.
    """
    from .nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
    from .nn.models import ContextualSvaraClassifier

    test = TEST_PERFORMER[raga]
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ContextualSvaraClassifier(EMBED_DIM, EMBED_DEPTH, len(RAGA_SVARAS[raga]))
    model.apply_lora(DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
    state = torch.load(WEIGHTS_DIR / f"{raga}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model.eval()
    model.to(device)
    encode = make_encoder(model, device, MIN_FRAMES, CONTEXT_DURATION, ENCODE_BATCH)

    classifier_path = CACHE_DIR / "classifier" / f"{raga}.joblib"
    markov_path = CACHE_DIR / "markov" / f"{raga}.npz"
    classifier = trigram = bigram = unigram = None
    if classifier_path.exists() and markov_path.exists():
        import joblib

        candidate = joblib.load(classifier_path)
        if candidate.named_steps["standardscaler"].n_features_in_ == CONTEXT_DIM:
            classifier = candidate
            stored = np.load(markov_path)
            trigram = stored["trigram"]
            bigram = stored["bigram"]
            unigram = stored["unigram"]
    if classifier is None:
        classifier, trigram, bigram, unigram = train_classifier(
            raga, encode, args, rng, tonics
        )
    return encode, classifier, trigram, bigram, unigram


def train_classifier(
    raga: str, encode, args: Namespace, rng: random.Random, tonics: dict
) -> tuple:
    """Fit the svara classifier and transition model on the training performances."""
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    test = TEST_PERFORMER[raga]
    training = [
        (performer, frames_for(raga, performer, tonics))
        for performer in sorted(
            pitch.stem
            for pitch in (DATA_DIR / "pitch_tracks" / raga).glob("*.tsv")
        )
        if performer != test
    ]
    features: list = []
    labels: list = []
    combined_features: list = []
    counts, bigrams, unigrams = Counter(), Counter(), Counter()
    for performer, item in training:
        times = item["times"].numpy()
        contour = item["contour"].numpy()
        begins, finishes, annotations = read_annotations(
            DATA_DIR / "annotations" / raga / f"{performer}_{raga}.tsv"
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
        performer, item = training[attempt % len(training)]
        attempt += 1
        if attempt > 20 * len(training) and not none_features:
            raise SystemExit(
                f"{raga}: no annotated coverage found in training performances"
            )
        times = item["times"].numpy()
        contour = item["contour"].numpy()
        begins, finishes, _ = read_annotations(
            DATA_DIR / "annotations" / raga / f"{performer}_{raga}.tsv"
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
    classifier_path = CACHE_DIR / "classifier" / f"{raga}.joblib"
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
    markov_path = CACHE_DIR / "markov" / f"{raga}.npz"
    markov_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(markov_path, trigram=trigram, bigram=bigram, unigram=unigram)
    return classifier, trigram, bigram, unigram


def transcribe_notes(
    raga: str,
    held: dict,
    regions: list[tuple[int, int]],
    beats: np.ndarray,
    beat_frames: float,
    args: Namespace,
    rng: random.Random,
    tonics: dict,
) -> list[tuple[int, int, int]]:
    """The reported variant's notes, from the cache; the genetic search runs on a miss."""
    cache = CACHE_DIR / "genetic" / raga / f"{TEST_PERFORMER[raga]}_{VARIANT}.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=False)
        if stored.get("regions") == regions and stored.get("cost") == list(COST):
            return stored["notes"]
    encode, classifier, trigram, bigram, unigram = raga_resources(
        raga, args, rng, tonics
    )
    times = held["times"].numpy()
    contour = held["contour"].numpy()

    def embed_fn(spans):
        return encode(times, contour, frame_spans_to_seconds(spans, times))

    optimiser = GeneticOptimiser(
        embed_fn, classifier, trigram, bigram, unigram, SVARAS, args, rng
    )
    notes = optimiser.transcribe(
        contour, regions, beats, beat_frames, COST[0], COST[1], COST[2], COST[3]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"notes": notes, "regions": regions, "cost": list(COST)}, cache)
    return notes


def emit_table_1(scores: dict[str, float]) -> None:
    """The note metrics table: F1 / precision / recall per note criterion."""
    print("**Table 1. Note metrics**\n")
    print("|  | COn | COnP | COnOff | COnPOff |")
    print("|---|---|---|---|---|")
    for label, key in (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")):
        cells = " | ".join(
            f"{scores[f'{criterion}_{key}']:.3f}"
            for criterion in ("con", "conp", "conoff", "conpoff")
        )
        print(f"| {label} | {cells} |")
    print()


def emit_table_2(rows: list[tuple]) -> None:
    """The longest-phrases table: transcription against annotation."""
    print(f"**Table 2. Top {PHRASE_COUNT} longest phrases**\n")
    print("| # | Transcription | Annotation | Lev Norm | LCS | LCS Norm | AOR |")
    print("|---|---|---|---|---|---|---|")
    for index, (transcription, annotation, lev, lcs, lcs_norm, aor) in enumerate(
        rows, start=1
    ):
        print(
            f"| {index} | {transcription} | {annotation} | "
            f"{lev:.3f} | {lcs} | {lcs_norm:.3f} | {aor:.3f} |"
        )
    print()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    tonics = yaml.safe_load((DATA_DIR / "tonics.yaml").read_text())

    for raga in RAGAS:
        test = TEST_PERFORMER[raga]
        held = frames_for(raga, test, tonics)
        times = held["times"].numpy()
        begins, finishes, annotations = read_annotations(
            DATA_DIR / "annotations" / raga / f"{test}_{raga}.tsv"
        )
        coverage = merge_coverage(begins, finishes)
        regions = coverage_frames(coverage, times)

        beat_times = read_beats(TRACKS_DIR / raga / f"{test}.tsv")
        window = (
            float(np.median(np.diff(beat_times.numpy()))) / 2.0
            if len(beat_times) > 1
            else 0.1
        )
        beats, beat_frames, _ = beat_spacing(beat_times, held["times"])

        notes = transcribe_notes(
            raga, held, regions, beats, beat_frames, args, rng, tonics
        )

        reference = reference_events(times, begins, finishes, annotations)
        estimated = in_coverage(notes_to_events(times, notes), coverage)
        scores = note_metrics(reference, estimated, window)
        phrases = phrase_rows(
            merge_coverage(begins, finishes, 0.5),
            reference,
            estimated,
            window,
            count=PHRASE_COUNT,
        )

        print(f"### {raga} ({test})\n")
        emit_table_1(scores)
        emit_table_2(phrases)


if __name__ == "__main__":
    main()

from __future__ import annotations

import gc
import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

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

import hmm  # noqa: F401
from genetic import SVARAS, genetic_transcribe
from plots import plot_transcription
from preprocess import (
    build_frames,
    coverage_frames,
    frame_spans_to_seconds,
    make_encoder,
    merge_coverage,
    read_annotations,
    sample_none_spans,
)
from rise.data.corpora import RAGA_SVARAS
from rise.nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from rise.nn.models import ContextualSvaraClassifier

console = Console(stderr=True)

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

PLOT_SECONDS = 15.0
EMBED_DIM = 48
EMBED_DEPTH = 5
CONTEXT_DIM = EMBED_DIM * 3


def main() -> None:
    root = Path(__file__).resolve().parents[1]
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

    data_dir = root / "data"
    cache_dir = root / ".cache" / "datasets"
    results_dir = root / "results"

    console.print("[bold]TRANSVARA[/bold]")
    table = Table()
    table.add_column("option")
    table.add_column("value")
    for key, value in {
        "cost": "L = (alpha * Lsvara + beta * Ltrans) / (alpha + beta)",
        "alpha": args.alpha,
        "beta": args.beta,
        "pm / pc / pd / pa": f"{args.pm} / {args.pc} / {args.pd} / {args.pa}",
        "ga population": args.ga_population,
        "ga iterations": args.ga_iterations,
        "dmin / dmax / smax": "half beat / two beats / one beat (TCN)",
        "model": "RISE fine-tuned contextual classifier (0.5 s melodic context)",
        "match window": "half beat (paper)",
        "sample rate": args.sample_rate,
        "seed": args.seed,
        "device": device,
    }.items():
        table.add_row(key, str(value))
    console.print(table)

    console.print("\n[bold]Data[/bold]")
    tonics = yaml.safe_load((data_dir / "tonics.yaml").read_text())

    progress = Progress(console=console)
    task = progress.add_task(
        "data",
        total=sum(
            len(list((data_dir / "pitch_tracks" / raga).glob("*.tsv")))
            for raga in RAGAS
        ),
    )
    progress.start()
    performances = []
    for raga in RAGAS:
        for pitch_path in sorted((data_dir / "pitch_tracks" / raga).glob("*.tsv")):
            performer = pitch_path.stem
            progress.update(task, description=f"{raga}/{performer}")
            frames_cache = cache_dir / "frames" / raga / f"{performer}.pt"
            if frames_cache.exists():
                stored = torch.load(frames_cache, map_location="cpu", weights_only=False)
            else:
                stored = build_frames(data_dir, raga, performer, tonics)
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
    available = {raga: set() for raga in RAGAS}
    for performance in performances:
        available[performance["raga"]].add(performance["performer"])
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
    )
    console.print(
        f"Loaded compIAM tcn_carnatic beat tracker from {beat_tracker.model_path}"
    )

    beats_dir = root / ".cache" / "beats"
    progress = Progress(console=console)
    task = progress.add_task("beats", total=len(performances))
    progress.start()
    for performance in performances:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        beats_cache = beats_dir / raga / f"{performer}.pt"
        if beats_cache.exists():
            beats = torch.load(
                beats_cache, map_location="cpu", weights_only=False
            )["beats"]
        else:
            source = data_dir / "audio" / raga / f"{performer}.mp3"
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

    classifier_dir = root / ".cache" / "classifier"
    markov_dir = root / ".cache" / "markov"
    genetic_dir = root / ".cache" / "genetic"
    test_performances = [
        item
        for item in performances
        if item["performer"] == TEST_PERFORMER[item["raga"]]
    ]

    progress = Progress(console=console)
    task = progress.add_task("classifier", total=2 * len(RAGAS))
    progress.start()
    for raga in RAGAS:
        ragged = [item for item in performances if item["raga"] == raga]
        test = TEST_PERFORMER[raga]
        training = [item for item in ragged if item["performer"] != test]

        progress.update(task, description=f"{raga}: classifier")
        console.print(f"\n[bold]{raga}[/bold]: loading fine-tuned RISE model")
        model = ContextualSvaraClassifier(EMBED_DIM, EMBED_DEPTH, len(RAGA_SVARAS[raga]))
        model.apply_lora(DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
        state = torch.load(
            root / "RISE" / ".cache" / "runs" / f"classification_{raga}_pretrained" / "best.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        del state
        model.eval()
        model.to(device)
        encode = make_encoder(model, device)

        classifier_path = classifier_dir / f"{raga}.joblib"
        markov_path = markov_dir / f"{raga}.npz"
        classifier: Any
        trigram: np.ndarray
        bigram: np.ndarray
        unigram: np.ndarray
        cached = False
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
            features: list[np.ndarray] = []
            labels: list[int] = []
            counts: dict[tuple[int, int, int], int] = {}
            bigrams: dict[tuple[int, int], int] = {}
            unigrams: dict[int, int] = {}
            for item in training:
                times = item["times"].numpy()
                contour = item["contour"].numpy()
                begins, finishes, annotations = read_annotations(
                    data_dir / "annotations" / raga / f"{item['performer']}_{raga}.tsv"
                )
                sequence = []
                spans: list[tuple[float, float]] = []
                for begin, finish, annotation in zip(
                    begins, finishes, annotations, strict=True
                ):
                    label = annotation[0]
                    if label not in SVARAS:
                        continue
                    if (
                        int(np.searchsorted(times, finish))
                        - int(np.searchsorted(times, begin))
                        < 2
                    ):
                        continue
                    spans.append((float(begin), float(finish)))
                    labels.append(SVARAS.index(label))
                    sequence.append(SVARAS.index(label))
                features.extend(list(encode(times, contour, spans)))
                for index in range(len(sequence) - 2):
                    bigrams[(sequence[index], sequence[index + 1])] = (
                        bigrams.get((sequence[index], sequence[index + 1]), 0) + 1
                    )
                    counts[
                        (sequence[index], sequence[index + 1], sequence[index + 2])
                    ] = counts.get(
                        (sequence[index], sequence[index + 1], sequence[index + 2]), 0
                    ) + 1
                for value in sequence:
                    unigrams[value] = unigrams.get(value, 0) + 1
            annotated = len(features)
            none_features: list[np.ndarray] = []
            attempt = 0
            while len(none_features) < annotated:
                item = training[attempt % len(training)]
                attempt += 1
                if attempt > 20 * len(training) and not none_features:
                    raise SystemExit(
                        f"{raga}: no annotated coverage found in training performances"
                    )
                times = item["times"].numpy()
                contour = item["contour"].numpy()
                begins, finishes, _ = read_annotations(
                    data_dir / "annotations" / raga / f"{item['performer']}_{raga}.tsv"
                )
                spans = coverage_frames(merge_coverage(begins, finishes), times)
                if not spans:
                    continue
                sampled = sample_none_spans(
                    spans, max(annotated // len(training), 1), rng
                )
                none_features.extend(list(encode(times, contour, frame_spans_to_seconds(sampled, times))))
            none_features = none_features[:annotated]
            features.extend(none_features)
            labels.extend([len(SVARAS)] * len(none_features))
            x = np.asarray(features, dtype=np.float64)
            y = np.asarray(labels, dtype=np.int64)
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
            classifier.fit(x, y)
            classifier_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(classifier, classifier_path)

            classes = len(SVARAS) + 1
            trigram = np.zeros((classes, classes, classes))
            bigram = np.zeros((classes, classes))
            unigram = np.zeros(classes)
            for (first, second, third), value in counts.items():
                trigram[first, second, third] = value
            for (first, second), value in bigrams.items():
                bigram[first, second] = value
            for key, value in unigrams.items():
                unigram[key] = value
            unigram = (unigram + 1e-3) / (unigram.sum() + 1e-3 * classes)
            markov_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(markov_path, trigram=trigram, bigram=bigram, unigram=unigram)
            console.print(
                f"{raga}: {annotated} annotated + {len(none_features)} none segments, "
                f"trained on {len(training)} performances (test {test})"
            )

        progress.advance(task)
        progress.update(task, description=f"{raga}: genetic")

        held = next(item for item in test_performances if item["raga"] == raga)
        cache = genetic_dir / raga / f"{test}.pt"
        times = held["times"].numpy()
        contour = held["contour"].numpy()
        begins, finishes, _ = read_annotations(
            data_dir / "annotations" / raga / f"{test}_{raga}.tsv"
        )
        regions = coverage_frames(merge_coverage(begins, finishes), times)
        notes: list[tuple[int, int, int]]
        cached = None
        if cache.exists():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            if stored.get("regions") == regions and stored.get("cost") == [
                args.alpha,
                args.beta,
            ]:
                notes = stored["notes"]
                cached = len(notes)
                console.print(f"{raga}/{test}: cached transcription ({cached} notes)")
        if cached is None:
            beat_times = held["beats"].numpy()
            beat_frames = (
                float(np.median(np.diff(beat_times)))
                / float(held["times"][1] - held["times"][0])
                if len(beat_times) > 1
                else 50.0
            )

            def embed_fn(
                spans: list[tuple[int, int]],
                times: np.ndarray = times,
                contour: np.ndarray = contour,
                encode: Any = encode,
            ) -> np.ndarray:
                return encode(
                    times, contour, frame_spans_to_seconds(spans, times)
                )

            console.print(f"{raga}/{test}: transcribing {len(regions)} annotated regions")
            notes = genetic_transcribe(
                contour,
                regions,
                embed_fn,
                classifier,
                trigram,
                bigram,
                unigram,
                beat_frames,
                args.alpha,
                args.beta,
                args.ga_population,
                args.ga_iterations,
                args.pm,
                args.pc,
                args.pd,
                args.pa,
                rng,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "notes": notes,
                    "regions": regions,
                    "cost": [args.alpha, args.beta],
                },
                cache,
            )
        held["notes"] = notes
        progress.advance(task)

        del model, encode
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    progress.stop()
    console.print("\n[bold]Evaluation[/bold]")
    progress = Progress(console=console)
    task = progress.add_task("evaluation", total=len(test_performances))
    progress.start()
    rows = []
    aggregated: dict[str, dict[str, list[float]]] = {}
    for performance in test_performances:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        times = performance["times"].numpy()
        step = float(times[1] - times[0]) if len(times) > 1 else 0.0
        estimated = [
            (float(times[start]), float(times[stop - 1]) + step, SVARAS[label])
            for start, stop, label in performance["notes"]
        ]

        begins, finishes, annotations = read_annotations(
            data_dir / "annotations" / raga / f"{performer}_{raga}.tsv"
        )
        reference = [
            (float(begins[index]), float(finishes[index]), annotations[index][0])
            for index in range(len(annotations))
            if annotations[index][0] in SVARAS
            and np.any((times >= begins[index]) & (times < finishes[index]))
        ]
        coverage = merge_coverage(begins, finishes)
        estimated = [
            note
            for note in estimated
            if any(note[0] >= begin and note[0] < finish for begin, finish in coverage)
        ]
        performance["reference"] = reference
        performance["estimated"] = estimated
        performance["coverage"] = coverage

        beat_times = performance["beats"].numpy()
        window = (
            float(np.median(np.diff(beat_times))) / 2.0 if len(beat_times) > 1 else 0.1
        )
        metrics = paper_metrics(reference, estimated, window)
        performance["metrics"] = metrics
        for key, value in metrics.items():
            aggregated.setdefault(raga, {}).setdefault(key, []).append(value)
        rows.append(
            (
                raga,
                performer,
                f"{metrics['con_f1']:.3f}",
                f"{metrics['conp_f1']:.3f}",
                f"{metrics['conoff_f1']:.3f}",
                f"{metrics['conpoff_f1']:.3f}",
                f"{metrics['aor']:.3f}",
                f"{metrics['levenshtein_norm']:.3f}",
                f"{metrics['lcs_norm']:.3f}",
            )
        )
        progress.advance(task)

    progress.stop()

    means = {
        raga: {key: float(np.mean(values)) for key, values in aggregated[raga].items()}
        for raga in RAGAS
    }
    means["all"] = {
        key: float(np.mean([value for raga in RAGAS for value in aggregated[raga][key]]))
        for key in aggregated[RAGAS[0]]
    }

    table = Table(title="Transcription (COn / COnP / COnOff / COnPOff)")
    for column in (
        "raga",
        "performer",
        "COn F",
        "COnP F",
        "COnOff F",
        "COnPOff F",
        "AOR",
        "Lev norm",
        "LCS norm",
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)

    table = Table(title="Mean (paper metrics)")
    table.add_column("raga")
    for column in (
        "COn F",
        "COn P",
        "COn R",
        "COnP F",
        "COnOff F",
        "COnPOff F",
        "AOR",
        "Lev norm",
        "LCS norm",
    ):
        table.add_column(column, justify="right")
    mean_keys = (
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
    for raga in [*RAGAS, "all"]:
        values = means[raga]
        table.add_row(raga, *(f"{values[key]:.3f}" for key in mean_keys))
    console.print(table)

    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "mean": means["all"],
                "per_raga": {
                    raga: {
                        "mean": means[raga],
                        "per_performance": [
                            {
                                "performer": item["performer"],
                                "metrics": item["metrics"],
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

    metrics_path = results_dir / "metrics.csv"
    pd.DataFrame(
        rows,
        columns=[
            "raga",
            "performer",
            "COn F",
            "COnP F",
            "COnOff F",
            "COnPOff F",
            "AOR",
            "Lev norm",
            "LCS norm",
        ],
    ).to_csv(metrics_path, index=False)
    console.print(f"table written to {metrics_path}")

    mean_path = results_dir / "mean.csv"
    pd.DataFrame(
        [
            (raga, key.replace("_", " "), value)
            for raga in [*RAGAS, "all"]
            for key, value in sorted(means[raga].items())
        ],
        columns=["raga", "metric", "value"],
    ).to_csv(mean_path, index=False)
    console.print(f"table written to {mean_path}")

    console.print("\n[bold]Plots[/bold]")
    plots_dir = root / "plots"
    for performance in test_performances:
        raga = performance["raga"]
        performer = performance["performer"]
        times = performance["times"].numpy()
        coverage = performance["coverage"]
        region = plot_region(coverage, times)
        signal, _ = librosa.load(
            data_dir / "audio" / raga / f"{performer}.mp3",
            sr=args.sample_rate,
            mono=True,
        )
        transcription_path = plots_dir / raga / f"{performer}_transcription.png"
        transcription_path.parent.mkdir(parents=True, exist_ok=True)
        plot_transcription(
            signal,
            args.sample_rate,
            [
                ("ground truth", performance["reference"]),
                ("genetic algorithm", performance["estimated"]),
            ],
            region,
            f"transcription ({performer})",
            transcription_path,
            coverage,
        )
        console.print(f"figure written to {transcription_path}")
        del signal


def lcs_length(first: str, second: str) -> int:
    previous = [0] * (len(second) + 1)
    for character in first:
        current = [0]
        for index, other in enumerate(second):
            if character == other:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[-1], previous[index + 1]))
        previous = current
    return previous[-1]


def match_counts(
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    window: float,
) -> tuple[int, int, int, int]:
    ref_intervals = np.array([[note[0], note[1]] for note in reference], dtype=float)
    est_intervals = np.array([[note[0], note[1]] for note in estimated], dtype=float)
    if not len(ref_intervals) or not len(est_intervals):
        return 0, 0, 0, 0
    ref_onsets = ref_intervals[:, 0]
    est_onsets = est_intervals[:, 0]
    matches = mir_eval.util.match_events(ref_onsets, est_onsets, window)
    onset = len(matches)
    pitch = sum(reference[i][2] == estimated[j][2] for i, j in matches)
    offset = sum(
        abs(reference[i][1] - estimated[j][1]) <= window for i, j in matches
    )
    both = sum(
        reference[i][2] == estimated[j][2]
        and abs(reference[i][1] - estimated[j][1]) <= window
        for i, j in matches
    )
    return onset, pitch, offset, both


def prf(count: int, estimated: int, reference: int) -> tuple[float, float, float]:
    precision = count / estimated if estimated else 0.0
    recall = count / reference if reference else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f, precision, recall


def paper_metrics(
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    window: float,
) -> dict[str, float]:
    ref_intervals = np.array(
        sorted([[note[0], note[1]] for note in reference]), dtype=float
    ).reshape(-1, 2)
    est_intervals = np.array(
        sorted([[note[0], note[1]] for note in estimated]), dtype=float
    ).reshape(-1, 2)
    ref_sorted = [reference[index] for index in np.argsort(ref_intervals[:, 0])] if len(ref_intervals) else []
    est_sorted = [estimated[index] for index in np.argsort(est_intervals[:, 0])] if len(est_intervals) else []

    onset, pitch, offset, both = match_counts(ref_sorted, est_sorted, window)
    metrics: dict[str, float] = {}
    for key, count in (
        ("con", onset),
        ("conp", pitch),
        ("conoff", offset),
        ("conpoff", both),
    ):
        f, p, r = prf(count, len(est_sorted), len(ref_sorted))
        metrics[f"{key}_f1"] = f
        metrics[f"{key}_precision"] = p
        metrics[f"{key}_recall"] = r

    ious = []
    if ref_sorted and est_sorted:
        ref_array = np.array([[note[0], note[1]] for note in ref_sorted])
        est_array = np.array([[note[0], note[1]] for note in est_sorted])
        matches = mir_eval.util.match_events(
            ref_array[:, 0], est_array[:, 0], window
        )
        for i, j in matches:
            intersection = max(
                0.0, min(ref_array[i, 1], est_array[j, 1]) - max(ref_array[i, 0], est_array[j, 0])
            )
            union = max(ref_array[i, 1], est_array[j, 1]) - min(
                ref_array[i, 0], est_array[j, 0]
            )
            ious.append(intersection / union if union > 0 else 0.0)
    metrics["aor"] = float(np.mean(ious)) if ious else 0.0

    ref_labels = "".join(note[2] for note in ref_sorted)
    est_labels = "".join(note[2] for note in est_sorted)
    distance = Levenshtein.distance(ref_labels, est_labels)
    lcs = lcs_length(ref_labels, est_labels)
    metrics["levenshtein"] = float(distance)
    metrics["levenshtein_norm"] = distance / max(len(ref_labels), 1)
    metrics["lcs"] = float(lcs)
    metrics["lcs_norm"] = lcs / max(len(ref_labels), 1)
    metrics["num_reference_notes"] = float(len(ref_sorted))
    metrics["num_estimated_notes"] = float(len(est_sorted))
    return metrics


def plot_region(
    coverage: list[list[float]], times: np.ndarray
) -> tuple[float, float]:
    duration = float(times[-1]) if len(times) else 0.0
    candidates = [span for span in coverage if span[1] - span[0] >= PLOT_SECONDS]
    if candidates:
        begin, finish = candidates[
            random.randrange(len(candidates))
        ]
        start = random.uniform(begin, finish - PLOT_SECONDS)
        return start, start + PLOT_SECONDS
    if coverage:
        longest = max(coverage, key=lambda span: span[1] - span[0])
        return longest[0], min(longest[1], longest[0] + PLOT_SECONDS)
    return 0.0, min(duration, PLOT_SECONDS)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TRANSVARA")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
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

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
from scipy.interpolate import UnivariateSpline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RISE" / "src"))

from compiam.utils.dbn import HiddenMarkovModel
from genetic import SVARAS, genetic_transcribe, segment_embeddings, voiced_regions
from plots import (
    plot_alignment,
    plot_confusion,
    plot_events,
    plot_transcription,
)
from rise.nn.models import SvaraEmbedder


def dbn_viterbi(self, log_densities):
    num_frames = len(log_densities)
    if num_frames == 0:
        return np.empty(0, dtype=np.int64), -np.inf
    frames = np.asarray(log_densities, dtype=np.float64)
    pointers = self.observation_pointers
    previous = np.full(self.num_states, -np.log(self.num_states))
    current = np.empty(self.num_states)
    backtrack = np.empty((num_frames, len(self._multi)), dtype=np.int32)
    has_multi = len(self._multi) > 0
    for frame in range(num_frames):
        if self._shift:
            current[1:] = previous[:-1]
        else:
            current[self._single] = (
                previous[self._single_prev] + self._single_log_prob
            )
        if has_multi:
            candidates = previous[self._multi_prev] + self._multi_log_prob
            best = candidates.argmax(axis=1)
            current[self._multi] = candidates[self._multi_rows, best]
            backtrack[frame] = self._multi_prev[self._multi_rows, best]
        current += frames[frame, pointers]
        previous, current = current, previous
    state = int(np.argmax(previous))
    log_probability = previous[state]
    if np.isinf(log_probability):
        return np.empty(0, dtype=np.int64), log_probability
    single_lookup = np.zeros(self.num_states, dtype=np.int64)
    single_lookup[self._single] = self._single_prev
    path = np.empty(num_frames, dtype=np.int64)
    for frame in range(num_frames - 1, -1, -1):
        path[frame] = state
        multi = self._multi_index[state]
        state = backtrack[frame, multi] if multi >= 0 else single_lookup[state]
    return path, log_probability


HiddenMarkovModel.viterbi = dbn_viterbi

console = Console()

RAGAS: tuple[str, ...] = (
    "abhogi",
    "begada",
    "kalyani",
    "mohanam",
    "sahana",
    "saveri",
    "sri",
)

PLOT_SECONDS = 15.0
EMBED_DIM = 48
EMBED_DEPTH = 5
EMBED_CHUNK_FRAMES = 8192
EMBED_CONTEXT = 50


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
        "cost": "alpha * Lsvara + beta * Ltrans - gamma * Llength",
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma (short note bias)": args.gamma,
        "pm / pc / pd / pa": f"{args.pm} / {args.pc} / {args.pd} / {args.pa}",
        "ga population": args.ga_population,
        "ga iterations": args.ga_iterations,
        "dmin / dmax / smax": "half beat / two beats / one beat (TCN)",
        "match window": args.match_window,
        "sample rate": args.sample_rate,
        "seed": args.seed,
        "device": device,
    }.items():
        table.add_row(key, str(value))
    console.print(table)

    console.print("\n[bold]Data[/bold]")
    tonics = yaml.safe_load((data_dir / "tonics.yaml").read_text())

    progress = Progress()
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
                    "voicing": stored["voicing"].to(torch.float32),
                }
            )
            progress.advance(task)

    progress.stop()
    console.print(f"{len(performances)} performances")

    console.print("\n[bold]Non-silent regions[/bold]")
    for performance in performances:
        performance["regions"] = voiced_regions(performance["voicing"].numpy())
    console.print(
        f"{sum(len(performance['regions']) for performance in performances)} regions"
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
    progress = Progress()
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

    console.print("\n[bold]Svara embeddings[/bold]")
    embedder = SvaraEmbedder(EMBED_DIM, EMBED_DEPTH)
    encoder_state = torch.load(
        root / "RISE" / "checkpoints" / "encoder.pth",
        map_location="cpu",
        weights_only=True,
    )
    embedder.encoder.load_state_dict(encoder_state)
    del encoder_state
    embedder.eval()
    embedder.to(device)
    console.print(f"Pretrained RISE encoder loaded from RISE/checkpoints/encoder.pth")

    embeddings_dir = root / ".cache" / "embeddings"
    progress = Progress()
    task = progress.add_task("embeddings", total=len(performances))
    progress.start()
    for performance in performances:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        embeddings_cache = embeddings_dir / raga / f"{performer}.pt"
        if embeddings_cache.exists():
            performance["embeddings"] = torch.load(
                embeddings_cache, map_location="cpu", weights_only=False
            )["embeddings"].numpy()
        else:
            performance["embeddings"] = encode_track(
                embedder,
                performance["contour"],
                performance["voicing"],
                device,
            )
            embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"embeddings": torch.from_numpy(performance["embeddings"].astype(np.float16))},
                embeddings_cache,
            )
        progress.advance(task)

    progress.stop()
    del embedder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    console.print("\n[bold]Classifier and Markov model[/bold]")
    classifier_dir = root / ".cache" / "classifier"
    markov_dir = root / ".cache" / "markov"
    classifiers: dict[str, Any] = {}
    markov: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for raga in RAGAS:
        classifier_path = classifier_dir / f"{raga}.joblib"
        markov_path = markov_dir / f"{raga}.npz"
        if classifier_path.exists() and markov_path.exists():
            classifiers[raga] = joblib.load(classifier_path)
            stored = np.load(markov_path)
            markov[raga] = (stored["trigram"], stored["bigram"], stored["unigram"])
            console.print(f"{raga}: cached classifier and Markov model")
            continue
        features, labels = [], []
        counts: dict[tuple[int, int, int], int] = {}
        bigrams: dict[tuple[int, int], int] = {}
        unigrams: dict[int, int] = {}
        for performance in [item for item in performances if item["raga"] == raga]:
            times = performance["times"].numpy()
            step = float(times[1] - times[0]) if len(times) > 1 else 0.0
            annotations = pd.read_csv(
                data_dir / "annotations" / raga / f"{performance['performer']}_{raga}.tsv",
                sep="\t",
            )
            sequence = []
            boundaries = [0]
            for begin, finish, annotation in zip(
                annotations["Begin time"],
                annotations["End time"],
                annotations["Annotation"],
                strict=True,
            ):
                label = str(annotation)[0]
                if label not in SVARAS:
                    continue
                start_frame = int(np.searchsorted(times, parse_time(begin)))
                stop_frame = int(np.searchsorted(times, parse_time(finish)))
                if stop_frame - start_frame < 2:
                    continue
                boundaries[-1] = start_frame
                boundaries.append(stop_frame)
                labels.append(SVARAS.index(label))
                sequence.append(SVARAS.index(label))
            if len(boundaries) > 1:
                segment_features = segment_embeddings(
                    build_prefix(performance["embeddings"]),
                    np.asarray(boundaries, dtype=np.int64),
                    EMBED_CONTEXT,
                )
                features.extend(list(segment_features))
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
        none_features = sample_none_segments(
            [item for item in performances if item["raga"] == raga], len(features), rng
        )
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
        classifiers[raga] = classifier
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
        markov[raga] = (trigram, bigram, unigram)
        markov_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(markov_path, trigram=trigram, bigram=bigram, unigram=unigram)
        console.print(f"{raga}: classifier fitted on {len(y)} segments")

    console.print("\n[bold]Genetic transcription[/bold]")
    genetic_dir = root / ".cache" / "genetic"
    progress = Progress()
    task = progress.add_task("genetic", total=len(performances))
    progress.start()
    for performance in performances:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        cache = genetic_dir / raga / f"{performer}.pt"
        if cache.exists():
            notes = torch.load(cache, map_location="cpu", weights_only=False)["notes"]
        else:
            beat_times = performance["beats"].numpy()
            beat_frames = (
                float(np.median(np.diff(beat_times)))
                / float(performance["times"][1] - performance["times"][0])
                if len(beat_times) > 1
                else 50.0
            )
            trigram, bigram, unigram = markov[raga]
            notes = genetic_transcribe(
                performance["contour"].numpy(),
                performance["embeddings"],
                performance["regions"],
                classifiers[raga],
                trigram,
                bigram,
                unigram,
                beat_frames,
                EMBED_CONTEXT,
                args.alpha,
                args.beta,
                args.gamma,
                args.ga_population,
                args.ga_iterations,
                args.pm,
                args.pc,
                args.pd,
                args.pa,
                rng,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"notes": notes}, cache)
        performance["notes"] = notes
        progress.advance(task)

    progress.stop()
    for performance in performances:
        performance.pop("embeddings", None)
        gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    console.print("\n[bold]Evaluation[/bold]")
    progress = Progress()
    task = progress.add_task("evaluation", total=len(performances))
    progress.start()
    condition = "genetic transcription"
    rows = []
    aggregated: dict[str, dict[str, list[float]]] = {}
    confusion_total = {
        raga: np.zeros((len(SVARAS), len(SVARAS)), dtype=np.int64) for raga in RAGAS
    }
    alignment_rows = {raga: [] for raga in RAGAS}
    for performance in performances:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        times = performance["times"].numpy()
        step = float(times[1] - times[0]) if len(times) > 1 else 0.0
        estimated = [
            (float(times[start]), float(times[stop - 1]) + step, SVARAS[label])
            for start, stop, label in performance["notes"]
        ]

        annotations = pd.read_csv(
            data_dir / "annotations" / raga / f"{performer}_{raga}.tsv", sep="\t"
        )
        starts = [parse_time(value) for value in annotations["Begin time"]]
        ends = [parse_time(value) for value in annotations["End time"]]
        labels = [str(label)[0] for label in annotations["Annotation"]]
        reference = [
            (starts[index], ends[index], labels[index])
            for index in range(len(labels))
            if labels[index] in SVARAS
            and np.any((times >= starts[index]) & (times < ends[index]))
        ]
        coverage: list[list[float]] = []
        for begin, finish in zip(starts, ends, strict=True):
            if coverage and begin <= coverage[-1][1]:
                coverage[-1][1] = max(coverage[-1][1], finish)
            else:
                coverage.append([begin, finish])
        estimated = [
            note
            for note in estimated
            if any(note[0] >= begin and note[0] < finish for begin, finish in coverage)
        ]
        performance["reference"] = reference
        performance["estimated"] = estimated

        ref_intervals = np.array(
            sorted([[note[0], note[1]] for note in reference]), dtype=float
        ).reshape(-1, 2)
        ref_onsets = ref_intervals[:, 0] if len(ref_intervals) else np.zeros(0)

        est_intervals = np.array(
            sorted([[note[0], note[1]] for note in estimated]), dtype=float
        ).reshape(-1, 2)
        est_onsets = est_intervals[:, 0] if len(est_intervals) else np.zeros(0)

        if len(ref_intervals) and len(est_intervals):
            onset_p, onset_r, onset_f = mir_eval.transcription.onset_precision_recall_f1(
                ref_intervals, est_intervals, args.match_window
            )
            offset_p, offset_r, offset_f = mir_eval.transcription.offset_precision_recall_f1(
                ref_intervals, est_intervals
            )
            boundary_p, boundary_r, boundary_f = mir_eval.segment.detection(
                ref_intervals, est_intervals, args.match_window
            )
            pairs = mir_eval.util.match_events(ref_onsets, est_onsets, args.match_window)
        else:
            onset_p = onset_r = onset_f = 0.0
            offset_p = offset_r = offset_f = 0.0
            boundary_p = boundary_r = boundary_f = 0.0
            pairs = []

        y_true = [SVARAS.index(reference[i][2]) for i, _ in pairs]
        y_pred = [SVARAS.index(estimated[j][2]) for _, j in pairs]
        svara_f = (
            f1_score(
                y_true, y_pred, labels=range(len(SVARAS)), average="macro", zero_division=0.0
            )
            if pairs
            else 0.0
        )
        label_accuracy = (
            float(np.mean(np.array(y_true) == np.array(y_pred))) if pairs else 0.0
        )
        confusion_total[raga] += confusion_matrix(
            y_true, y_pred, labels=range(len(SVARAS))
        )

        ref_labels = "".join(note[2] for note in reference)
        est_labels = "".join(note[2] for note in estimated)
        distance = Levenshtein.distance(ref_labels, est_labels)
        accuracy = 1.0 - distance / max(len(ref_labels), len(est_labels), 1)

        metrics = {
            "onset_precision": float(onset_p),
            "onset_recall": float(onset_r),
            "onset_f_measure": float(onset_f),
            "offset_precision": float(offset_p),
            "offset_recall": float(offset_r),
            "offset_f_measure": float(offset_f),
            "boundary_precision": float(boundary_p),
            "boundary_recall": float(boundary_r),
            "boundary_f_measure": float(boundary_f),
            "svara_f_measure": float(svara_f),
            "label_accuracy": label_accuracy,
            "alignment_accuracy": float(accuracy),
            "num_reference_notes": float(len(reference)),
            "num_estimated_notes": float(len(estimated)),
        }
        for key, value in metrics.items():
            aggregated.setdefault(raga, {}).setdefault(key, []).append(value)
        rows.append(
            (
                raga,
                performer,
                f"{boundary_f:.3f}",
                f"{onset_f:.3f}",
                f"{offset_f:.3f}",
                f"{svara_f:.3f}",
                f"{label_accuracy:.3f}",
                f"{accuracy:.3f}",
            )
        )

        correct_ref = {i for i, j in pairs if reference[i][2] == estimated[j][2]}
        correct_est = {j for i, j in pairs if reference[i][2] == estimated[j][2]}
        alignment_rows[raga].append(
            (
                f"{raga}/{performer}",
                [
                    (note[0], note[1], note[2], index in correct_ref)
                    for index, note in enumerate(reference)
                ],
                [
                    (note[0], note[1], note[2], index in correct_est)
                    for index, note in enumerate(estimated)
                ],
            )
        )
        progress.advance(task)

    progress.stop()

    headline = (
        "boundary_f_measure",
        "onset_f_measure",
        "offset_f_measure",
        "svara_f_measure",
        "label_accuracy",
        "alignment_accuracy",
    )
    means = {
        raga: {key: float(np.mean(values)) for key, values in aggregated[raga].items()}
        for raga in RAGAS
    }
    means["all"] = {
        key: float(np.mean([value for raga in RAGAS for value in aggregated[raga][key]]))
        for key in aggregated[RAGAS[0]]
    }

    table = Table(title=f"Transcription ({condition})")
    for column in (
        "raga",
        "performer",
        "boundary F",
        "onset F",
        "offset F",
        "svara F",
        "label acc",
        "align acc",
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)

    table = Table(title=f"Mean ({condition})")
    table.add_column("raga")
    for column in (
        "boundary F",
        "onset F",
        "offset F",
        "svara F",
        "label acc",
        "align acc",
    ):
        table.add_column(column, justify="right")
    for raga in [*RAGAS, "all"]:
        values = means[raga]
        table.add_row(raga, *(f"{values[key]:.3f}" for key in headline))
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
                        "per_performance": [row for row in rows if row[0] == raga],
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
            "boundary F",
            "onset F",
            "offset F",
            "svara F",
            "label acc",
            "align acc",
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
    for raga in RAGAS:
        performance = next(item for item in performances if item["raga"] == raga)
        performer = performance["performer"]
        figure = plot_confusion(
            confusion_total[raga],
            list(SVARAS),
            f"Svara confusion ({raga})",
        )
        figure_path = plots_dir / raga / "confusion.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        console.print(f"figure written to {figure_path}")

        alignment_path = plots_dir / raga / "alignment.png"
        plot_alignment(alignment_rows[raga], alignment_path)
        console.print(f"figure written to {alignment_path}")

        times = performance["times"].numpy()
        duration = float(times[-1]) if len(times) else 0.0
        start = float(np.random.uniform(0, max(duration - PLOT_SECONDS, 0.0)))
        region = (start, min(start + PLOT_SECONDS, duration))
        signal, _ = librosa.load(
            data_dir / "audio" / raga / f"{performer}.mp3",
            sr=args.sample_rate,
            mono=True,
        )
        region_seconds = [
            (float(times[start_frame]), float(times[min(stop_frame, len(times) - 1)]))
            for start_frame, stop_frame in performance["regions"]
        ]
        transcription_path = plots_dir / raga / "transcription.png"
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
            region_seconds,
        )
        console.print(f"figure written to {transcription_path}")

        beats = torch.load(
            beats_dir / raga / f"{performer}.pt",
            map_location="cpu",
            weights_only=False,
        )["beats"].numpy()
        beats_path = plots_dir / raga / "beattrack.png"
        plot_events(
            signal,
            args.sample_rate,
            beats,
            f"TCN beat tracking ({performer})",
            region,
            "#2ca02c",
            beats_path,
        )
        console.print(f"figure written to {beats_path}")
        del signal, beats


def parse_time(value: str) -> float:
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TRANSVARA")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--pm", type=float, default=0.75)
    parser.add_argument("--pc", type=float, default=0.75)
    parser.add_argument("--pd", type=float, default=0.5)
    parser.add_argument("--pa", type=float, default=0.75)
    parser.add_argument("--ga-population", type=int, default=40)
    parser.add_argument("--ga-iterations", type=int, default=50)
    parser.add_argument("--match-window", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_frames(
    data_dir: Path, raga: str, performer: str, tonics: dict
) -> dict[str, torch.Tensor]:
    tonic = float(tonics[performer])
    track = pd.read_csv(
        data_dir / "pitch_tracks" / raga / f"{performer}.tsv",
        sep="\t",
        header=None,
        names=["time", "frequency"],
    )
    times = track["time"].to_numpy(dtype=float)
    frequency = track["frequency"].to_numpy(dtype=float)

    raw = frequency.copy()
    raw[raw == 0] = np.nan
    voicing = (~np.isnan(raw)).astype(float)

    contour = raw.copy()
    is_gap = np.isnan(contour)
    padded = np.concatenate(([False], is_gap, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        if stop - start > 0.02:
            continue
        contour[start:stop] = np.nan
    filled = pd.Series(contour).interpolate(method="linear").ffill().bfill().to_numpy()

    smoothed = np.full_like(filled, np.nan)
    voiced = np.where(~np.isnan(filled))[0]
    if voiced.size:
        for run in np.split(voiced, np.where(np.diff(voiced) > 1)[0] + 1):
            run_times, run_cents = times[run], filled[run]
            if run.size >= 4:
                low, high = run_cents.min(), run_cents.max()
                spline = UnivariateSpline(
                    run_times, (run_cents - low) / (high - low), s=0.5
                )
                smoothed[run] = spline(run_times) * (high - low) + low
            elif run.size > 1:
                smoothed[run] = np.interp(run_times, run_times, run_cents)

    cents = 1200.0 * np.log2(smoothed / tonic)
    cents = (cents + 4200.0) / 8400.0

    return {
        "tonic": tonic,
        "times": torch.from_numpy(times),
        "contour": torch.from_numpy(cents.astype(np.float32)),
        "voicing": torch.from_numpy(voicing.astype(np.float32)),
    }


def encode_track(
    embedder: SvaraEmbedder,
    contour: torch.Tensor,
    voicing: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    features = torch.stack(
        [torch.nan_to_num(contour, nan=0.0) * voicing, 1.0 - voicing], dim=-1
    ).transpose(0, 1)
    frames = features.shape[1]
    pooled: list[torch.Tensor] = []
    with torch.no_grad():
        for begin in range(0, frames, EMBED_CHUNK_FRAMES):
            chunk = features[:, begin : begin + EMBED_CHUNK_FRAMES]
            pooled.append(embedder.encoder(chunk.unsqueeze(0).to(device))[0].cpu())
    encoded = torch.cat(pooled, dim=1).numpy()
    positions = np.linspace(0, encoded.shape[1] - 1, frames)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = positions - left
    return (
        encoded[:, left].T * (1.0 - weight)[:, np.newaxis]
        + encoded[:, right].T * weight[:, np.newaxis]
    ).astype(np.float32)


def build_prefix(embeddings: np.ndarray) -> np.ndarray:
    prefix = np.zeros((embeddings.shape[1], embeddings.shape[0] + 1), dtype=np.float64)
    prefix[:, 1:] = np.cumsum(np.asarray(embeddings, dtype=np.float64), axis=0).T
    return prefix


def sample_none_segments(
    performances: list[dict], count: int, rng: random.Random
) -> list[np.ndarray]:
    spans: list[np.ndarray] = []
    for performance in performances:
        prefix = build_prefix(performance["embeddings"])
        for start, stop in performance["regions"]:
            if stop - start < 4:
                continue
            length = rng.randrange(4, 64)
            if stop - start <= length:
                continue
            begin = rng.randrange(start, stop - length)
            spans.extend(
                segment_embeddings(
                    prefix,
                    np.asarray([begin, begin + length], dtype=np.int64),
                    EMBED_CONTEXT,
                )
            )
            if len(spans) >= count:
                return spans
    return spans


if __name__ == "__main__":
    main()

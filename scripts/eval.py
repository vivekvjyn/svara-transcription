from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

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
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "GAME"))

from inference.me_infer import SegmentationEstimationInferenceModel
from lib.config.schema import InferenceConfig, ModelConfig
from plots import plot_alignment, plot_confusion
from tsvara.transcription import TranscriptionModel

console = Console()

SVARAS: tuple[str, ...] = ("S", "R", "G", "M", "P", "D", "N")

RAGAS: tuple[str, ...] = (
    "abhogi",
    "begada",
    "kalyani",
    "mohanam",
    "sahana",
    "saveri",
    "sri",
)

TEST_PERFORMERS: dict[str, str] = {
    "abhogi": "prasanna",
    "begada": "prasanna",
    "kalyani": "prasanna",
    "mohanam": "prasanna",
    "sahana": "vignesh",
    "saveri": "prasanna",
    "sri": "ramakrishnamurthy",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    args = parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    weights_dir = root / "model_weights"
    results_dir = root / "results"
    model_cfg = cfg["model"]
    num_classes = len(SVARAS) + 1

    console.print("[bold]tsvara eval[/bold]")
    table = Table()
    table.add_column("option")
    table.add_column("value")
    for key, value in {
        "min note frames": args.min_note_frames,
        "inference chunk": args.inference_chunk,
        "inference overlap": args.inference_overlap,
        "match window": args.match_window,
        "sample rate": args.sample_rate,
        "audio fps": args.audio_fps,
        "fft size": args.fft_size,
        "hop size": args.hop_size,
        "filter fmin": args.filter_fmin,
        "filter fmax": args.filter_fmax,
        "filter bands": args.filter_bands,
        "log epsilon": args.log_epsilon,
        "postprocess window": args.postprocess_window,
        "seed": args.seed,
        "device": device,
    }.items():
        table.add_row(key, str(value))
    console.print(table)

    hop = args.hop_size if args.hop_size else args.sample_rate // args.audio_fps
    mel_filters = librosa.filters.mel(
        sr=args.sample_rate,
        n_fft=args.fft_size,
        n_mels=args.filter_bands,
        fmin=args.filter_fmin,
        fmax=args.filter_fmax,
    ).T

    console.print("\n[bold]Data[/bold]")
    progress = Progress()
    task = progress.add_task("data", total=len(RAGAS))
    progress.start()
    test = []
    for raga in RAGAS:
        performer = TEST_PERFORMERS[raga]
        progress.update(task, description=f"{raga}/{performer}")
        frames_cache = cache_dir / "frames" / raga / f"{performer}.pt"
        if not frames_cache.exists():
            raise FileNotFoundError(frames_cache)
        stored = torch.load(frames_cache, map_location="cpu", weights_only=False)
        item = {
            "raga": raga,
            "performer": performer,
            "times": stored["times"],
            "contour": stored["contour"].to(torch.float32),
            "voicing": stored["voicing"].to(torch.float32),
        }

        source = data_dir / "audio" / raga / f"{performer}.mp3"
        signal, _ = librosa.load(source, sr=args.sample_rate, mono=True)
        audio_cache = cache_dir / "audio" / raga / f"{performer}.pt"
        if audio_cache.exists():
            features = torch.load(audio_cache, map_location="cpu", weights_only=True)
        else:
            magnitude = np.abs(
                librosa.stft(
                    signal,
                    n_fft=args.fft_size,
                    hop_length=hop,
                    win_length=args.fft_size,
                    window=np.hanning(args.fft_size),
                    center=False,
                )
            ).T
            features = torch.from_numpy(
                np.log(magnitude @ mel_filters + args.log_epsilon).astype(np.float32)
            )
            audio_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, audio_cache)

        item["waveform"] = torch.from_numpy(signal)
        item["sample_rate"] = args.sample_rate
        num_frames = len(item["times"])
        if features.shape[0] >= num_frames:
            item["audio"] = features[:num_frames]
        else:
            padding = features[-1:].expand(num_frames - features.shape[0], -1)
            item["audio"] = torch.cat([features, padding], dim=0)
        test.append(item)
        progress.advance(task)

    progress.stop()
    console.print(f"{len(test)} held-out test performances")

    model = TranscriptionModel(
        num_classes,
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        tcn_filters=model_cfg["tcn_filters"],
        tcn_kernel=model_cfg["tcn_kernel"],
        tcn_dilations=model_cfg["tcn_dilations"],
        dropout=model_cfg["dropout"],
        segmenter_depth=model_cfg["segmenter_depth"],
        boundary_std=model_cfg["boundary_std"],
        boundary_threshold=model_cfg["boundary_threshold"],
        boundary_radius=model_cfg["boundary_radius"],
        d3pm_steps=model_cfg["d3pm_steps"],
    ).to(device)

    console.print("\n[bold]Model[/bold]")
    model.apply_lora(
        model_cfg["lora_rank"], model_cfg["lora_alpha"], model_cfg["lora_dropout"]
    )
    checkpoints = {}
    for raga in RAGAS:
        path = weights_dir / "tsvara" / raga / "best.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoints[raga] = path
        console.print(f"Fine-tuned weights for {raga}: {path}")

    console.print("\n[bold]Game[/bold]")
    game_config = yaml.safe_load(
        (weights_dir / "game" / "game_pretrained_config.yaml").read_text()
    )
    game_model_config = ModelConfig.model_validate(game_config["model"], scope=0)
    game_inference_config = InferenceConfig.model_validate(
        game_config["inference"], scope=0
    )
    game_model_config.check(scope_mask=0)
    game_inference_config.check(scope_mask=0)
    game_model = SegmentationEstimationInferenceModel(
        game_model_config, game_inference_config
    )
    game_checkpoint = weights_dir / "game" / "game_pretrained.pt"
    game_state = torch.load(game_checkpoint, map_location="cpu", weights_only=True)
    game_model.load_state_dict(game_state["state_dict"], strict=True)
    game_model.eval()
    game_model.to(device)
    console.print(f"Loaded frozen GAME from {game_checkpoint}")

    console.print("\n[bold]Evaluation[/bold]")
    progress = Progress()
    task = progress.add_task("evaluation", total=len(test))
    progress.start()
    conditions = ("without TCN postprocessing", "with TCN postprocessing")
    rows = {condition: [] for condition in conditions}
    aggregated = {condition: {} for condition in conditions}
    confusion_total = {
        condition: {
            raga: np.zeros((len(SVARAS), len(SVARAS)), dtype=np.int64)
            for raga in RAGAS
        }
        for condition in conditions
    }
    alignment_rows = {raga: [] for raga in RAGAS}
    for performance in test:
        raga = performance["raga"]
        performer = performance["performer"]
        progress.update(task, description=f"{raga}/{performer}")
        model.load_state_dict(
            torch.load(checkpoints[raga], map_location=device, weights_only=True)
        )
        times = performance["times"].numpy()
        estimated, estimated_post = transcribe(
            model,
            game_model,
            performance,
            device,
            args.inference_chunk,
            args.inference_overlap,
            args.min_note_frames,
            args.postprocess_window,
        )

        annotations = pd.read_csv(
            data_dir / "annotations" / raga / f"{performer}_{raga}.tsv", sep="\t"
        )
        starts = []
        ends = []
        for begin, finish in zip(
            annotations["Begin time"], annotations["End time"], strict=True
        ):
            seconds = 0.0
            for part in str(begin).split(":"):
                seconds = seconds * 60 + float(part)
            starts.append(seconds)
            seconds = 0.0
            for part in str(finish).split(":"):
                seconds = seconds * 60 + float(part)
            ends.append(seconds)
        labels = [str(label)[0] for label in annotations["Annotation"]]
        reference = [
            (starts[index], ends[index], labels[index])
            for index in range(len(labels))
            if labels[index] in SVARAS
            and np.any((times >= starts[index]) & (times < ends[index]))
        ]

        ref_intervals = np.array(
            sorted([[note[0], note[1]] for note in reference]), dtype=float
        ).reshape(-1, 2)
        ref_onsets = ref_intervals[:, 0] if len(ref_intervals) else np.zeros(0)

        for condition, notes in zip(conditions, (estimated, estimated_post), strict=True):
            est_intervals = np.array(
                sorted([[note[0], note[1]] for note in notes]), dtype=float
            ).reshape(-1, 2)
            est_onsets = est_intervals[:, 0] if len(est_intervals) else np.zeros(0)

            if len(ref_intervals) and len(est_intervals):
                onset_p, onset_r, onset_f, _ = mir_eval.transcription.onset_precision_recall_f1(
                    ref_intervals, est_intervals, args.match_window
                )
                offset_p, offset_r, offset_f, _ = mir_eval.transcription.offset_precision_recall_f1(
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
            y_pred = [SVARAS.index(notes[j][2]) for _, j in pairs]
            svara_f = (
                f1_score(y_true, y_pred, labels=range(len(SVARAS)), average="macro", zero_division=0.0)
                if pairs
                else 0.0
            )
            label_accuracy = (
                float(np.mean(np.array(y_true) == np.array(y_pred))) if pairs else 0.0
            )
            confusion_total[condition][raga] += confusion_matrix(
                y_true, y_pred, labels=range(len(SVARAS))
            )

            ref_labels = "".join(note[2] for note in reference)
            est_labels = "".join(note[2] for note in notes)
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
                "num_estimated_notes": float(len(notes)),
            }
            for key, value in metrics.items():
                aggregated[condition].setdefault(raga, {}).setdefault(key, []).append(value)
            rows[condition].append(
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

            if condition == conditions[1]:
                correct_ref = {i for i, j in pairs if reference[i][2] == notes[j][2]}
                correct_est = {j for i, j in pairs if reference[i][2] == notes[j][2]}
                alignment_rows[raga].append(
                    (
                        f"{raga}/{performer}",
                        [
                            (note[0], note[1], note[2], index in correct_ref)
                            for index, note in enumerate(reference)
                        ],
                        [
                            (note[0], note[1], note[2], index in correct_est)
                            for index, note in enumerate(notes)
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
    means = {}
    for condition in conditions:
        means[condition] = {}
        for raga in RAGAS:
            means[condition][raga] = {
                key: float(np.mean(values))
                for key, values in aggregated[condition][raga].items()
            }
        means[condition]["all"] = {
            key: float(
                np.mean(
                    [
                        value
                        for raga in RAGAS
                        for value in aggregated[condition][raga][key]
                    ]
                )
            )
            for key in means[condition][RAGAS[0]]
        }

    for condition in conditions:
        table = Table(title=f"Test transcription ({condition})")
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
        for row in rows[condition]:
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
            values = means[condition][raga]
            table.add_row(raga, *(f"{values[key]:.3f}" for key in headline))
        console.print(table)

    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                condition: {
                    "mean": means[condition]["all"],
                    "per_raga": {
                        raga: {
                            "mean": means[condition][raga],
                            "per_performance": [
                                row for row in rows[condition] if row[0] == raga
                            ],
                        }
                        for raga in RAGAS
                    },
                }
                for condition in conditions
            },
            indent=4,
        )
    )
    console.print(f"results written to {results_path}")

    metrics_path = results_dir / "metrics.csv"
    pd.DataFrame(
        [(condition,) + row for condition in conditions for row in rows[condition]],
        columns=[
            "condition",
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
            (condition, raga, key.replace("_", " "), value)
            for condition in conditions
            for raga in [*RAGAS, "all"]
            for key, value in sorted(means[condition][raga].items())
        ],
        columns=["condition", "raga", "metric", "value"],
    ).to_csv(mean_path, index=False)
    console.print(f"table written to {mean_path}")

    for raga in RAGAS:
        figure = plot_confusion(
            confusion_total[conditions[1]][raga],
            list(SVARAS),
            f"Svara confusion ({raga})",
        )
        figure_path = results_dir / f"confusion_{raga}.png"
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        console.print(f"figure written to {figure_path}")

        alignment_path = results_dir / f"alignment_{raga}.png"
        plot_alignment(alignment_rows[raga], alignment_path)
        console.print(f"figure written to {alignment_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate the tsvara transcription model")
    parser.add_argument("--min-note-frames", type=int, default=5)
    parser.add_argument("--inference-chunk", type=int, default=4096)
    parser.add_argument("--inference-overlap", type=int, default=512)
    parser.add_argument("--match-window", type=float, default=0.05)
    parser.add_argument("--postprocess-window", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--audio-fps", type=int, default=100)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=None)
    parser.add_argument("--filter-fmin", type=float, default=30.0)
    parser.add_argument("--filter-fmax", type=float, default=17000.0)
    parser.add_argument("--filter-bands", type=int, default=81)
    parser.add_argument("--log-epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def transcribe(
    model: TranscriptionModel,
    game_model: SegmentationEstimationInferenceModel,
    performance: dict,
    device: torch.device,
    chunk: int,
    overlap: int,
    min_frames: int,
    window: int,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    total = performance["times"].shape[0]
    inputs = np.stack(
        [performance["contour"].numpy(), performance["voicing"].numpy()]
    ).astype(np.float32)
    stride = max(chunk - overlap, 1)
    embedding = None
    counts = np.zeros(total, dtype=np.float32)

    for start in range(0, total, stride):
        stop = min(start + chunk, total)
        crop = torch.from_numpy(inputs[:, start:stop]).unsqueeze(0).to(device)
        features = model.encode(crop)[0].cpu().numpy()
        if embedding is None:
            embedding = np.zeros((features.shape[0], total), dtype=np.float32)
        embedding[:, start:stop] += features
        counts[start:stop] += 1.0
        if stop == total:
            break
    embedding = torch.from_numpy(embedding / counts).unsqueeze(0).to(device)

    times = performance["times"].numpy()
    spans, refined = segment(
        model, game_model, performance, times, device, min_frames, chunk, overlap, window
    )
    notes = classify(model, embedding, spans, times, SVARAS)
    notes_refined = classify(model, embedding, refined, times, SVARAS)
    return notes, notes_refined


@torch.no_grad()
def segment(
    model: TranscriptionModel,
    game_model: SegmentationEstimationInferenceModel,
    performance: dict,
    times: np.ndarray,
    device: torch.device,
    min_frames: int,
    chunk: int,
    overlap: int,
    window: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    frames = len(times)
    step = float(times[1] - times[0]) if frames > 1 else 0.0
    hop_samples = int(round(performance["sample_rate"] * step))
    waveform = performance["waveform"]
    boundary_flags = np.zeros(frames, dtype=bool)
    stride = max(chunk - overlap, 1)
    for start in range(0, frames, stride):
        stop = min(start + chunk, frames)
        if stop - start < 2:
            break
        crop = waveform[start * hop_samples : stop * hop_samples]
        crop = crop.unsqueeze(0).to(device)
        duration = torch.tensor(
            [[crop.shape[1] / performance["sample_rate"]]],
            dtype=torch.float32,
            device=device,
        )
        durations, _, _ = game_model(
            crop,
            duration,
            0.3,
            2,
            0.2,
            language=torch.zeros(1, dtype=torch.long, device=device),
            t=torch.arange(8, dtype=torch.float32, device=device) / 8,
        )
        ends = np.cumsum(durations[0].detach().cpu().numpy())
        for value in ends:
            frame = start + int(round(float(value) / step))
            if 0 < frame < frames:
                boundary_flags[frame] = True
        if stop == frames:
            break

    bounds = [0] + np.flatnonzero(boundary_flags).tolist() + [frames]
    spans = [
        (bounds[index], bounds[index + 1])
        for index in range(len(bounds) - 1)
        if bounds[index + 1] - bounds[index] >= min_frames
    ]

    audio_embedding = model.encode_audio(performance["audio"].unsqueeze(0).to(device))[0]
    delta = (audio_embedding[:, 1:] - audio_embedding[:, :-1]).norm(dim=0)
    delta = delta.detach().cpu().numpy()
    refined = [bounds[0]]
    for value in bounds[1:-1]:
        lo = max(value - 1 - window, 0)
        hi = min(value - 1 + window + 1, len(delta))
        if hi <= lo:
            refined.append(value)
            continue
        refined.append(int(np.argmax(delta[lo:hi])) + 1)
    refined.append(frames)
    refined = sorted(set(refined))
    refined_spans = [
        (refined[index], refined[index + 1])
        for index in range(len(refined) - 1)
        if refined[index + 1] - refined[index] >= min_frames
    ]
    return spans, refined_spans


@torch.no_grad()
def classify(
    model: TranscriptionModel,
    embedding: torch.Tensor,
    spans: list[tuple[int, int]],
    times: np.ndarray,
    alphabet: tuple[str, ...],
) -> list[tuple[float, float, str]]:
    was_training = model.training
    model.eval()
    regions = torch.zeros(embedding.shape[2], dtype=torch.long, device=embedding.device)
    for region, (start, stop) in enumerate(spans, start=1):
        regions[start:stop] = region
    logits = model.classifier(model.pool(embedding.transpose(1, 2), regions.unsqueeze(0)))
    if was_training:
        model.train()

    labels = logits[0].argmax(dim=-1).cpu().numpy()
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    notes = []
    for (start, stop), label in zip(spans, labels, strict=True):
        if int(label) >= len(alphabet):
            continue
        notes.append((float(times[start]), float(times[stop - 1]) + step, alphabet[int(label)]))
    return notes


if __name__ == "__main__":
    main()

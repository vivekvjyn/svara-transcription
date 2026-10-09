from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from train import RAGAS, SVARAS, TEST_PERFORMERS
from tsvara.transcription import TranscriptionModel
from utils import (
    VarnamCorpus,
    Performance,
    apply_plot_style,
    artifact,
    banner,
    build_performances,
    detail,
    load_config,
    metrics_table,
    parameters_table,
    read_elan_annotations,
    resolve_device,
    rule,
    seed_everything,
    split_performances,
    svara_of,
)

SVARA_DISPLAY_NAMES: dict[str, str] = {
    "S": "Sa",
    "R": "Ri",
    "G": "Ga",
    "M": "Ma",
    "P": "Pa",
    "D": "Dha",
    "N": "Ni",
}

FPS = 100.0


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    svara: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def decode_notes(
    outputs: dict[str, np.ndarray],
    performance: Performance,
    svaras: tuple[str, ...],
    fps: float,
    *,
    min_frames: int = 5,
) -> list[Note]:
    regions = outputs["regions"]
    labels = outputs["region_logits"].argmax(axis=-1)
    first = outputs["region_first_frame"]
    times = performance.times
    frames = performance.num_frames

    notes: list[Note] = []
    for region in range(1, int(regions.max()) + 1):
        start = int(first[region])
        if start >= frames:
            continue
        stop = int(first[region + 1]) if region + 1 < len(first) else frames
        if stop - start < min_frames:
            continue
        label = int(labels[region - 1])
        if label >= len(svaras):
            continue
        notes.append(Note(times[start], times[stop - 1] + 1 / fps, svaras[label]))
    return notes


def reference_notes(
    performance: Performance,
    corpus: VarnamCorpus,
    svaras: tuple[str, ...],
) -> list[Note]:
    annotations = read_elan_annotations(corpus.annotation_path(performance.raga, performance.performer))
    notes: list[Note] = []
    for row in annotations.itertuples(index=False):
        svara = svara_of(row.label)
        if svara not in svaras:
            continue
        if not np.any((performance.times >= row.start) & (performance.times < row.end)):
            continue
        notes.append(Note(row.start, row.end, svara))
    return notes


def note_intervals(notes: list[Note]) -> npt.NDArray[np.float64]:
    if not notes:
        return np.zeros((0, 2), dtype=float)
    return np.array([[note.start, note.end] for note in notes], dtype=float)


def match_times(reference: list[float], estimated: list[float], window: float) -> tuple[int, int, int]:
    ref = np.sort(np.asarray(reference, dtype=float))
    est = np.sort(np.asarray(estimated, dtype=float))
    matched = 0
    cursor = 0
    claimed = np.zeros(len(ref), dtype=bool)
    for event in est:
        while cursor < len(ref) and ref[cursor] < event - window:
            cursor += 1
        if cursor < len(ref) and abs(ref[cursor] - event) <= window and not claimed[cursor]:
            claimed[cursor] = True
            matched += 1
            cursor += 1
    return matched, len(ref), len(est)


def _prf(matched: int, reference: int, estimated: int) -> dict[str, float]:
    precision = matched / estimated if estimated else 0.0
    recall = matched / reference if reference else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f_measure": f}


def boundary_metrics(reference: list[Note], estimated: list[Note], window: float) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, times in (
        ("onset", ([note.start for note in reference], [note.start for note in estimated])),
        ("offset", ([note.end for note in reference], [note.end for note in estimated])),
    ):
        matched, ref_count, est_count = match_times(times[0], times[1], window)
        for key, value in _prf(matched, ref_count, est_count).items():
            metrics[f"{name}_{key}"] = value

    ref_intervals = note_intervals(reference)
    est_intervals = note_intervals(estimated)
    if len(ref_intervals) and len(est_intervals):
        from mir_eval.segment import detection

        precision, recall, f = detection(ref_intervals, est_intervals, window)
    else:
        precision = recall = f = 0.0
    metrics["boundary_precision"] = float(precision)
    metrics["boundary_recall"] = float(recall)
    metrics["boundary_f_measure"] = float(f)
    return metrics


def match_notes(
    reference: list[Note], estimated: list[Note], window: float
) -> list[tuple[int, int]]:
    order = np.argsort([note.start for note in estimated])
    ref_onsets = [note.start for note in reference]
    pairs: list[tuple[int, int]] = []

    claimed = np.zeros(len(reference), dtype=bool)
    positions = np.argsort(ref_onsets)
    cursor = 0
    for est_index in order:
        onset = estimated[est_index].start
        while cursor < len(positions) and ref_onsets[positions[cursor]] < onset - window:
            cursor += 1
        if cursor < len(positions):
            ref_index = int(positions[cursor])
            if abs(ref_onsets[ref_index] - onset) <= window and not claimed[ref_index]:
                claimed[ref_index] = True
                pairs.append((ref_index, int(est_index)))
                cursor += 1
    return pairs


def classification_metrics(
    reference: list[Note], estimated: list[Note], alphabet: tuple[str, ...], window: float
) -> dict[str, float]:
    pairs = match_notes(reference, estimated, window)
    correct = [(ref, est) for ref, est in pairs if reference[ref].svara == estimated[est].svara]

    ref_counts = {svara: 0 for svara in alphabet}
    est_counts = {svara: 0 for svara in alphabet}
    correct_counts = {svara: 0 for svara in alphabet}
    for note in reference:
        if note.svara in ref_counts:
            ref_counts[note.svara] += 1
    for note in estimated:
        if note.svara in est_counts:
            est_counts[note.svara] += 1
    for ref, est in correct:
        correct_counts[reference[ref].svara] += 1

    per_svara: dict[str, dict[str, float]] = {}
    for svara in alphabet:
        precision = correct_counts[svara] / est_counts[svara] if est_counts[svara] else 0.0
        recall = correct_counts[svara] / ref_counts[svara] if ref_counts[svara] else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_svara[svara] = {"precision": precision, "recall": recall, "f_measure": f}

    n_correct = len(correct)
    precision = n_correct / len(estimated) if estimated else 0.0
    recall = n_correct / len(reference) if reference else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "svara_precision": precision,
        "svara_recall": recall,
        "svara_f_measure": f,
        "svara_macro_f": float(np.mean([values["f_measure"] for values in per_svara.values()])),
        "matched_notes": float(len(pairs)),
        "correct_notes": float(n_correct),
        "label_accuracy": n_correct / len(pairs) if pairs else 0.0,
        "num_reference_notes": float(len(reference)),
        "num_estimated_notes": float(len(estimated)),
    }
    metrics.update(
        {f"{svara}_{key}": value for svara, values in per_svara.items() for key, value in values.items()}
    )
    return metrics


def confusion(
    reference: list[Note], estimated: list[Note], alphabet: tuple[str, ...], window: float
) -> npt.NDArray[np.int64]:
    pairs = match_notes(reference, estimated, window)
    index = {svara: position for position, svara in enumerate(alphabet)}
    true = [index[reference[ref].svara] for ref, _ in pairs]
    predicted = [index[estimated[est].svara] for _, est in pairs]
    return confusion_matrix(true, predicted, labels=range(len(alphabet)))


def plot_confusion(matrix: npt.NDArray[np.int64], labels: list[str], title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(
        matrix,
        annot=matrix.astype(int),
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Notes"},
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        square=True,
        annot_kws={"size": 8},
        ax=axis,
    )
    axis.set_xlabel("Transcribed")
    axis.set_ylabel("Annotated")
    axis.set_title(title)
    axis.tick_params(length=0)
    figure.tight_layout()
    return figure


def load_model_weights(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


@torch.no_grad()
def predict_performance(
    model: TranscriptionModel,
    performance: Performance,
    device: torch.device,
    *,
    chunk: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.eval()

    total = performance.num_frames
    inputs = np.stack([performance.contour, performance.voicing]).astype(np.float32)
    stride = max(chunk - overlap, 1)
    embedding = None
    counts = np.zeros(total, dtype=np.float32)

    for start in range(0, total, stride):
        stop = min(start + chunk, total)
        window = torch.from_numpy(inputs[:, start:stop]).unsqueeze(0).to(device)
        features = model.encode(window)[0].cpu().numpy()
        if embedding is None:
            embedding = np.zeros((features.shape[0], total), dtype=np.float32)
        embedding[:, start:stop] += features
        counts[start:stop] += 1.0
        if stop == total:
            break
    embedding = torch.from_numpy(embedding / counts).unsqueeze(0).to(device)

    outputs = model.transcribe_embedding(embedding)
    if was_training:
        model.train()
    return {
        key: value[0].cpu().numpy() if value.ndim > 1 else value.cpu().numpy()
        for key, value in outputs.items()
    }


def parse_args(cfg: dict) -> Namespace:
    evaluation = cfg["evaluation"]
    parser = ArgumentParser(description="Evaluate a fine-tuned tsvara transcription model")
    parser.add_argument("--run-tag", type=str, default="joint")
    parser.add_argument("--min-note-frames", type=int, default=evaluation["min_note_frames"])
    parser.add_argument("--inference-chunk", type=int, default=evaluation["inference_chunk"])
    parser.add_argument("--inference-overlap", type=int, default=evaluation["inference_overlap"])
    parser.add_argument("--match-window", type=float, default=evaluation["match_window"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    cfg = load_config()
    args = parse_args(cfg)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    frames_dir = root / ".cache" / "datasets" / "frames"
    weights_dir = root / "weights"
    results_dir = root / "results"
    model_cfg = cfg["model"]
    num_classes = len(SVARAS) + 1

    banner("tsvara eval")
    parameters_table(
        {
            "run tag": args.run_tag,
            "min note frames": args.min_note_frames,
            "inference chunk": args.inference_chunk,
            "inference overlap": args.inference_overlap,
            "match window": args.match_window,
            "seed": args.seed,
            "device": device,
        }
    )

    corpus = VarnamCorpus(
        pitch_track_dir=data_dir / "pitch_tracks",
        annotation_dir=data_dir / "annotations",
        audio_dir=data_dir / "audio",
        tonics_file=data_dir / "tonics.yaml",
    )

    rule("Data")
    performances = build_performances(corpus, RAGAS, SVARAS, frames_dir)
    test = []
    for raga in RAGAS:
        subset = [p for p in performances if p.raga == raga]
        _, test_part = split_performances(subset, TEST_PERFORMERS[raga])
        test.extend(test_part)
    detail(f"{len(test)} held-out test performances")

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
    model.apply_lora(
        model_cfg["lora_rank"], model_cfg["lora_alpha"], model_cfg["lora_dropout"]
    )

    rule("Model")
    checkpoint = weights_dir / "runs" / args.run_tag / "best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    load_model_weights(model, checkpoint)
    detail(f"Loaded fine-tuned weights from {checkpoint}")

    rule("Evaluation")
    rows = []
    aggregated: dict[str, list[float]] = {}
    confusion_total = np.zeros((len(SVARAS), len(SVARAS)), dtype=np.int64)
    for performance in test:
        outputs = predict_performance(
            model,
            performance,
            device,
            chunk=args.inference_chunk,
            overlap=args.inference_overlap,
        )
        estimated = decode_notes(
            outputs, performance, SVARAS, FPS, min_frames=args.min_note_frames
        )
        reference = reference_notes(performance, corpus, SVARAS)

        boundary = boundary_metrics(reference, estimated, args.match_window)
        classification = classification_metrics(reference, estimated, SVARAS, args.match_window)
        confusion_total += confusion(reference, estimated, SVARAS, args.match_window)

        for key, value in {**boundary, **classification}.items():
            aggregated.setdefault(key, []).append(value)
        rows.append(
            (
                f"{performance.raga}/{performance.performer}",
                f"{boundary['boundary_f_measure']:.3f}",
                f"{classification['svara_f_measure']:.3f}",
                f"{classification['label_accuracy']:.3f}",
            )
        )

    metrics_table(
        "Test transcription",
        ("performance", "boundary F", "svara F", "label acc"),
        rows,
    )
    means = {key: float(np.mean(values)) for key, values in aggregated.items()}
    metrics_table(
        "Mean",
        ("metric", "value"),
        [(key.replace("_", " "), f"{value:.3f}") for key, value in sorted(means.items())],
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{args.run_tag}.json"
    results_path.write_text(json.dumps({"mean": means, "per_performance": rows}, indent=2))
    artifact("Results", results_path)

    metrics_path = results_dir / f"{args.run_tag}_metrics.csv"
    pd.DataFrame(
        rows, columns=["performance", "boundary F", "svara F", "label acc"]
    ).to_csv(metrics_path, index=False)
    artifact("Table", metrics_path)

    mean_path = results_dir / f"{args.run_tag}_mean.csv"
    pd.DataFrame(
        [{"metric": key.replace("_", " "), "value": value} for key, value in sorted(means.items())]
    ).to_csv(mean_path, index=False)
    artifact("Table", mean_path)

    figure = plot_confusion(
        confusion_total,
        [SVARA_DISPLAY_NAMES[s] for s in SVARAS],
        "Svara confusion",
    )
    figure_path = results_dir / f"{args.run_tag}_confusion.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    artifact("Figure", figure_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def read_pitch_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column ``time`` / ``frequency`` pitch track."""
    track = pd.read_csv(path, sep="\t", header=None, names=["time", "frequency"])
    return track["time"].to_numpy(dtype=float), track["frequency"].to_numpy(dtype=float)


def tonic_for(tonics_path: Path, performer: str) -> float:
    """Return the performer's tonic from the tonics file."""
    tonics = yaml.safe_load(tonics_path.read_text()) if tonics_path.exists() else {}
    if performer not in tonics:
        raise SystemExit(f"no tonic for {performer} in {tonics_path}")
    return float(tonics[performer])


def build_frames(pitch_path: Path, tonic: float) -> dict[str, torch.Tensor]:
    """Clean a raw pitch track into the normalised contour the model consumes."""
    from rise.data.preprocessing import prepare_contour

    times, frequency = read_pitch_track(pitch_path)
    contour = prepare_contour(times, frequency, tonic, 0.5, 7)
    return {
        "tonic": tonic,
        "times": torch.from_numpy(times),
        "contour": torch.from_numpy(((contour + 4200.0) / 8400.0).astype(np.float32)),
    }


def beat_spacing(
    beats: torch.Tensor, times: torch.Tensor
) -> tuple[np.ndarray, float, float]:
    """Frame index of every beat, plus one beat in frames and in seconds."""
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


def read_annotations(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read ``start time`` / ``end time`` / ``label`` annotation rows.

    Timestamps are ``HH:MM:SS.mmm`` and labels are reduced to their svara
    symbol's first letter, upper-cased, so ``"sa"`` and ``"S^"`` both give
    ``"S"``.
    """
    annotations = pd.read_csv(path, sep="\t")
    begins = pd.to_timedelta(annotations["start time"]).dt.total_seconds().to_numpy()
    finishes = pd.to_timedelta(annotations["end time"]).dt.total_seconds().to_numpy()
    labels = [str(label).strip()[0].upper() for label in annotations["label"]]
    return begins, finishes, labels


def merge_coverage(
    begins: np.ndarray, finishes: np.ndarray, gap: float = 0.0
) -> list[list[float]]:
    """Collapse sorted annotation spans into runs separated by more than ``gap`` seconds."""
    coverage: list[list[float]] = []
    for begin, finish in zip(begins.tolist(), finishes.tolist(), strict=True):
        if coverage and begin <= coverage[-1][1] + gap:
            coverage[-1][1] = max(coverage[-1][1], finish)
        else:
            coverage.append([begin, finish])
    return coverage


def coverage_frames(coverage: list[list[float]], times: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    for begin, finish in coverage:
        start = int(np.searchsorted(times, begin))
        stop = int(np.searchsorted(times, finish))
        if stop - start >= 5:
            spans.append((start, stop))
    return spans


def sample_none_spans(
    spans: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]]:
    sampled: list[tuple[int, int]] = []
    for _ in range(count):
        start, stop = rng.choice(spans)
        if stop - start < 4:
            continue
        length = rng.randrange(4, min(64, stop - start))
        begin = rng.randrange(start, stop - length)
        sampled.append((begin, begin + length))
    return sampled


def _window(
    times: np.ndarray,
    contour: np.ndarray,
    begin: float,
    end: float,
    min_frames: int,
) -> np.ndarray:
    values = contour[(times >= begin) & (times < end)]
    if values.size == 0:
        return np.full(min_frames, np.nan)
    if values.size < min_frames:
        values = np.pad(values, (0, min_frames - values.size), mode="edge")
    return values


def _batch(windows: list[np.ndarray]) -> torch.Tensor:
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(window, dtype=torch.float32) for window in windows],
        batch_first=True,
    ).unsqueeze(1)
    silence = torch.isnan(padded).to(torch.float32)
    return torch.cat([torch.nan_to_num(padded, nan=0.0), silence], dim=1)


def make_encoder(
    model, device, min_frames: int, context_duration: float, encode_batch: int
):
    def encode(times: np.ndarray, contour: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
        outputs = []
        for begin in range(0, len(spans), encode_batch):
            chunk = spans[begin : begin + encode_batch]
            prec = _batch(
                [
                    _window(
                        times, contour, start - context_duration, start, min_frames
                    )
                    for start, _ in chunk
                ]
            )
            curr = _batch(
                [
                    _window(times, contour, start, stop, min_frames)
                    for start, stop in chunk
                ]
            )
            succ = _batch(
                [
                    _window(times, contour, stop, stop + context_duration, min_frames)
                    for _, stop in chunk
                ]
            )
            with torch.no_grad():
                encoded = model.encode(
                    prec.to(device), curr.to(device), succ.to(device)
                )
            outputs.append(encoded.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros((0, 144), dtype=np.float32)

    return encode


def frame_spans_to_seconds(
    spans: list[tuple[int, int]], times: np.ndarray
) -> list[tuple[float, float]]:
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    return [
        (float(times[start]), float(times[stop]) if stop < len(times) else float(times[-1]) + step)
        for start, stop in spans
    ]

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONTEXT_DURATION = 0.5
ENCODE_BATCH = 128
MIN_FRAMES = 16


def build_frames(
    data_dir: Path, raga: str, performer: str, tonics: dict
) -> dict[str, torch.Tensor]:
    from rise.data.preprocessing import prepare_contour

    tonic = float(tonics[performer])
    track = pd.read_csv(
        data_dir / "pitch_tracks" / raga / f"{performer}.tsv",
        sep="\t",
        header=None,
        names=["time", "frequency"],
    )
    times = track["time"].to_numpy(dtype=float)
    frequency = track["frequency"].to_numpy(dtype=float)
    contour = prepare_contour(times, frequency, tonic, 0.5, 0.02)
    return {
        "tonic": tonic,
        "times": torch.from_numpy(times),
        "contour": torch.from_numpy(((contour + 4200.0) / 8400.0).astype(np.float32)),
    }


def read_annotations(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    annotations = pd.read_csv(path, sep="\t")
    begins = pd.to_timedelta(annotations["Begin time"]).dt.total_seconds().to_numpy()
    finishes = pd.to_timedelta(annotations["End time"]).dt.total_seconds().to_numpy()
    labels = [str(label) for label in annotations["Annotation"]]
    return begins, finishes, labels


def merge_coverage(begins: np.ndarray, finishes: np.ndarray) -> list[list[float]]:
    coverage: list[list[float]] = []
    for begin, finish in zip(begins.tolist(), finishes.tolist(), strict=True):
        if coverage and begin <= coverage[-1][1]:
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


def _window(times: np.ndarray, contour: np.ndarray, begin: float, end: float) -> np.ndarray:
    values = contour[(times >= begin) & (times < end)]
    if values.size == 0:
        return np.full(MIN_FRAMES, np.nan)
    if values.size < MIN_FRAMES:
        values = np.pad(values, (0, MIN_FRAMES - values.size), mode="edge")
    return values


def _batch(windows: list[np.ndarray]) -> torch.Tensor:
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(window, dtype=torch.float32) for window in windows],
        batch_first=True,
    ).unsqueeze(1)
    silence = torch.isnan(padded).to(torch.float32)
    return torch.cat([torch.nan_to_num(padded, nan=0.0), silence], dim=1)


def make_encoder(model, device):
    def encode(times: np.ndarray, contour: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
        outputs = []
        for begin in range(0, len(spans), ENCODE_BATCH):
            chunk = spans[begin : begin + ENCODE_BATCH]
            prec = _batch(
                [_window(times, contour, start - CONTEXT_DURATION, start) for start, _ in chunk]
            )
            curr = _batch([_window(times, contour, start, stop) for start, stop in chunk])
            succ = _batch(
                [_window(times, contour, stop, stop + CONTEXT_DURATION) for _, stop in chunk]
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

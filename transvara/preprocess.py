from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.interpolate import UnivariateSpline


def hz_to_cents(
    frequency: npt.ArrayLike, tonic: float
) -> npt.NDArray[np.float64]:
    return 1200 * np.log2(np.asarray(frequency, dtype=float) / tonic)


def interpolate_gaps(
    values: npt.ArrayLike,
    gap_value: float = np.nan,
    max_gap: float = 0.02,
    protected_indices: Iterable[int] | None = None,
) -> npt.NDArray[np.float64]:
    contour = np.asarray(values, dtype=float).copy()
    protected = set(protected_indices or ())
    is_gap = np.isnan(contour) if np.isnan(gap_value) else contour == gap_value

    padded = np.concatenate(([False], np.asarray(is_gap, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(
        edges[::2].tolist(), edges[1::2].tolist(), strict=True
    ):
        if stop - start > max_gap:
            continue
        if protected & set(range(start, stop)):
            continue
        contour[start:stop] = np.nan

    return pd.Series(contour).interpolate(method="linear").ffill().bfill().to_numpy()


def smooth_contour(
    times: npt.ArrayLike,
    cents: npt.ArrayLike,
    smoothing_factor: float = 0.5,
    min_points: int = 4,
) -> npt.NDArray[np.float64]:
    times = np.asarray(times, dtype=float)
    cents = np.asarray(cents, dtype=float)
    smoothed = np.full_like(cents, np.nan)

    voiced = np.where(~pd.isna(times) & ~pd.isna(cents))[0]
    if voiced.size == 0:
        return smoothed

    for run in np.split(voiced, np.where(np.diff(voiced) > 1)[0] + 1):
        run_times, run_cents = times[run], cents[run]
        if run.size >= min_points:
            low, high = run_cents.min(), run_cents.max()
            spline = UnivariateSpline(run_times, (run_cents - low) / (high - low), s=smoothing_factor)
            smoothed[run] = spline(run_times) * (high - low) + low
        elif run.size > 1:
            smoothed[run] = np.interp(run_times, run_times, run_cents)

    return smoothed


def prepare_contour(
    times: npt.NDArray[np.float64],
    frequency: npt.NDArray[np.float64],
    tonic: float,
    smoothing_factor: float,
    interpolation_gap: float,
) -> npt.NDArray[np.float64]:
    contour = np.asarray(frequency, dtype=float).copy()
    contour[contour == 0] = np.nan
    contour = interpolate_gaps(contour, np.nan, interpolation_gap)
    contour = smooth_contour(times, contour, smoothing_factor)
    return hz_to_cents(contour, tonic)


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


def merge_coverage(
    begins: np.ndarray, finishes: np.ndarray, gap: float = 0.0
) -> list[list[float]]:
    coverage: list[list[float]] = []
    for begin, finish in zip(begins.tolist(), finishes.tolist(), strict=True):
        if coverage and begin <= coverage[-1][1] + gap:
            coverage[-1][1] = max(coverage[-1][1], finish)
        else:
            coverage.append([begin, finish])
    return coverage


def read_beats(path: Path) -> torch.Tensor:
    track = pd.read_csv(path, sep="\t", header=None, names=["time", "beat"])
    return torch.from_numpy(track["time"].to_numpy(dtype=float))


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

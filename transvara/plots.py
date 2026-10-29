from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_region(
    coverage: list[list[float]], times: np.ndarray, seconds: float
) -> tuple[float, float]:
    duration = float(times[-1]) if len(times) else 0.0
    candidates = [span for span in coverage if span[1] - span[0] >= seconds]
    if candidates:
        begin, finish = candidates[random.randrange(len(candidates))]
        start = random.uniform(begin, finish - seconds)
        return start, start + seconds
    if coverage:
        longest = max(coverage, key=lambda span: span[1] - span[0])
        return longest[0], min(longest[1], longest[0] + seconds)
    return 0.0, min(duration, seconds)


def apply_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _svara_colour(svara: str):
    order = "SRGMPDN"
    index = order.find(svara.upper())
    if index < 0:
        index = len(order)
    return plt.cm.tab20((index % 20 + 0.5) / 20)


def plot_transcription(
    waveform: npt.NDArray[np.float32],
    sample_rate: int,
    conditions: list[tuple[str, list[tuple[float, float, str]]]],
    region: tuple[float, float],
    title: str,
    path: Path,
    vocal: list[tuple[float, float]] | None = None,
) -> None:
    apply_plot_style()
    start, stop = region
    first = int(max(start, 0.0) * sample_rate)
    last = int(min(stop, len(waveform) / sample_rate) * sample_rate)
    times = np.arange(first, last) / sample_rate
    clip = waveform[first:last]
    stride = max(len(times) // 20000, 1)
    times = times[::stride]
    clip = clip[::stride]
    peak = float(np.max(np.abs(clip))) if clip.size else 1.0
    peak = max(peak, 1e-6)
    band = 0.22 * peak
    top = peak + 0.5 * peak

    figure, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(10, 2.3 * len(conditions)),
        sharex=True,
        squeeze=False,
    )
    for axis, (condition, intervals) in zip(axes[:, 0], conditions):
        if vocal:
            for vocal_start, vocal_end in vocal:
                if vocal_end <= start or vocal_start >= stop:
                    continue
                axis.axvspan(
                    max(vocal_start, start),
                    min(vocal_end, stop),
                    facecolor="#1f77b4",
                    alpha=0.12,
                    linewidth=0,
                )
        axis.plot(times, clip, color="#cccccc", linewidth=0.4)
        for index, (note_start, note_end, svara) in enumerate(intervals):
            if note_end <= start or note_start >= stop:
                continue
            begin = max(note_start, start)
            finish = min(note_end, stop)
            colour = _svara_colour(svara)
            mask = (times >= begin) & (times < finish)
            axis.plot(times[mask], clip[mask], color=colour, linewidth=0.4)
            axis.broken_barh(
                [(begin + 0.01, max(finish - begin - 0.02, 0.02))],
                (top, band),
                facecolor=colour,
                edgecolor="black",
                linewidth=0.7,
            )
            axis.text(
                (begin + finish) / 2,
                top + band / 2,
                svara,
                ha="center",
                va="center",
                fontsize=7,
            )
        axis.set_ylim(-1.3 * peak, top + 1.6 * band)
        axis.set_yticks([])
        axis.set_title(condition, loc="left", fontsize=9)
    axes[0, 0].set_xlim(start, stop)
    axes[-1, 0].set_xlabel("time (s)")
    figure.suptitle(title, x=0.01, ha="left", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
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
    import matplotlib.pyplot as plt

    order = "SRGMPDN"
    index = order.find(svara.upper())
    if index < 0:
        index = len(order)
    return plt.cm.tab20((index % 20 + 0.5) / 20)


def plot_transcription(
    times: npt.NDArray[np.float64],
    contour: npt.NDArray[np.float32],
    conditions: list[tuple[str, list[tuple[float, float, str]]]],
    region: tuple[float, float],
    title: str,
    path: Path,
    vocal: list[tuple[float, float]] | None = None,
    beats: npt.NDArray[np.float64] | None = None,
    downbeats: npt.NDArray[np.float64] | None = None,
) -> None:
    """Plot the pitch contour of a region with each note coloured by svara.

    The vertical axis is cents relative to the tonic, labelled with the
    *bhairavi* svarasthanas — every svara at its equal-tempered position
    (S 0, R 100, G 300, M 500, P 700, D 800, N 1000 cents) repeated per
    octave, so a glide reads against the scale it lands on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    apply_plot_style()
    start, stop = region
    sthanas = (
        (0, "S"),
        (100, "R"),
        (300, "G"),
        (500, "M"),
        (700, "P"),
        (800, "D"),
        (1000, "N"),
    )
    cents = np.asarray(contour, dtype=float) * 8400.0 - 4200.0
    window = cents[(times >= start) & (times < stop)]
    voiced = window[np.isfinite(window)]
    if voiced.size:
        low, high = float(voiced.min()), float(voiced.max())
    else:
        low, high = 0.0, 1200.0
    pad = max((high - low) * 0.12, 100.0)
    low, high = low - pad, high + pad

    ticks, labels = [], []
    for octave in range(int(np.floor(low / 1200)), int(np.floor(high / 1200)) + 1):
        for offset, letter in sthanas:
            position = octave * 1200 + offset
            if low <= position <= high:
                ticks.append(position)
                labels.append(letter)

    figure, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(10, 2.6 * len(conditions)),
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
        axis.plot(times, cents, color="#999999", linewidth=0.6, zorder=1)
        if beats is not None:
            for beat in beats[(beats >= start) & (beats <= stop)]:
                axis.axvline(
                    beat,
                    color="#666666",
                    linewidth=0.6,
                    linestyle=(0, (3, 2)),
                    alpha=0.8,
                    zorder=1.5,
                )
        if downbeats is not None:
            for downbeat in downbeats[(downbeats >= start) & (downbeats <= stop)]:
                axis.axvline(
                    downbeat,
                    color="#333333",
                    linewidth=1.0,
                    zorder=1.6,
                )
        bars = blended_transform_factory(axis.transData, axis.transAxes)
        for note_start, note_end, svara in intervals:
            if note_end <= start or note_start >= stop:
                continue
            begin = max(note_start, start)
            finish = min(note_end, stop)
            colour = _svara_colour(svara)
            mask = (times >= begin) & (times < finish)
            axis.plot(times[mask], cents[mask], color=colour, linewidth=1.4, zorder=3)
            axis.broken_barh(
                [(begin, max(finish - begin, 0.02))],
                (0.9, 0.07),
                transform=bars,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.7,
            )
            axis.text(
                (begin + finish) / 2,
                0.935,
                svara,
                transform=bars,
                ha="center",
                va="center",
                fontsize=7,
            )
        axis.set_ylim(low, high)
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
        axis.set_title(condition, loc="left", fontsize=9)
    axes[0, 0].set_xlim(start, stop)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("svara")
    figure.suptitle(title, x=0.01, ha="left", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_windows(
    coverage: list[list[float]], low: float = 15.0, high: float = 20.0
) -> list[tuple[float, float]]:
    """Cut every phrase into non-overlapping windows of 15 to 20 seconds.

    A phrase of ``length`` seconds is divided into the count of windows whose
    span lands inside ``low``..``high`` and is closest to their midpoint, so
    no two figures share a moment of the performance and a phrase shorter
    than ``low`` is plotted whole instead of being cut up.
    """
    windows: list[tuple[float, float]] = []
    target = 0.5 * (low + high)
    for begin, finish in coverage:
        length = finish - begin
        best_count = 1
        best_key: tuple[int, float] | None = None
        for count in range(1, max(int(np.ceil(length / low)), 1) + 2):
            chunk = length / count
            key = (0 if low <= chunk <= high else 1, abs(chunk - target))
            if best_key is None or key < best_key:
                best_count, best_key = count, key
        step = length / best_count
        windows.extend(
            (begin + index * step, begin + (index + 1) * step)
            for index in range(best_count)
        )
    return windows

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def apply_plot_style() -> None:
    import matplotlib

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


def save_history_plot(scores: list[float], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    epochs = np.arange(1, len(scores) + 1)
    best = int(np.argmax(scores)) + 1
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    axis.plot(epochs, scores, color="#1f77b4", label="Validation score")
    axis.plot(
        best,
        scores[best - 1],
        "o",
        color="#d62728",
        markersize=6,
        label=f"Best epoch {best}",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation score")
    axis.set_title("Fine-tuning curve")
    axis.set_xlim(1, max(len(scores), 1))
    axis.grid(axis="y", linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


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


def plot_alignment(rows: list[tuple[str, list, list]], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_plot_style()
    palette = {
        "S": "#1f77b4",
        "R": "#ff7f0e",
        "G": "#2ca02c",
        "M": "#9467bd",
        "P": "#17becf",
        "D": "#8c564b",
        "N": "#e377c2",
    }
    figure, axes = plt.subplots(
        len(rows), 1, figsize=(8.5, 1.7 * len(rows)), squeeze=False
    )
    for axis, (title, reference, estimated) in zip(axes[:, 0], rows):
        for start, end, svara, correct in reference:
            axis.broken_barh(
                [(start, max(end - start, 0.02))],
                (0.55, 0.35),
                facecolor=palette.get(svara, "#999999") if correct else "#d62728",
                edgecolor="white",
                linewidth=0.3,
            )
        for start, end, svara, correct in estimated:
            axis.broken_barh(
                [(start, max(end - start, 0.02))],
                (0.1, 0.35),
                facecolor=palette.get(svara, "#999999") if correct else "#d62728",
                edgecolor="white",
                linewidth=0.3,
            )
        axis.set_ylim(0, 1)
        axis.set_yticks([0.72, 0.27], ["ref", "est"])
        axis.set_title(title, loc="left", fontsize=8)
        axis.set_xlabel("time (s)")
        axis.tick_params(axis="x", labelsize=8)
    handles = [Patch(facecolor=color, label=svara) for svara, color in palette.items()]
    handles.append(Patch(facecolor="#d62728", label="error"))
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

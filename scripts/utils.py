from __future__ import annotations

import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme
from scipy.interpolate import UnivariateSpline

console = Console(
    theme=Theme(
        {
            "title": "bold cyan",
            "stage": "bold cyan",
            "metric": "bold green",
            "delta": "yellow",
            "path": "dim",
            "detail": "dim",
            "warning": "bold yellow",
            "error": "bold red",
        }
    ),
    highlight=False,
)


def rule(title: str) -> None:
    console.rule(f"[title]{title}[/title]")


def banner(text: str) -> None:
    console.print(Panel(f"[stage]{text}[/stage]", expand=False, border_style="cyan"))


def detail(text: str) -> None:
    console.print(f"  [detail]{text}[/detail]")


def warn(text: str) -> None:
    console.print(f"[warning]![/warning] {text}")


def artifact(kind: str, path) -> None:
    console.print(f"  [detail]{kind} written to[/detail] [path]{path}[/path]")


def progress(*, transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def metrics_table(title: str, columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    table = Table(title=title, title_style="title", header_style="bold")
    table.add_column(columns[0])
    for column in columns[1:]:
        table.add_column(column, justify="right", style="metric")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def parameters_table(parameters: Mapping[str, object]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="detail")
    table.add_column(justify="right")
    for key, value in sorted(parameters.items()):
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)


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


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path | None = None) -> dict:
    target = path if path is not None else Path(__file__).resolve().parents[1] / "config.yaml"
    with target.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@dataclass(frozen=True)
class VarnamCorpus:
    pitch_track_dir: Path
    annotation_dir: Path
    audio_dir: Path
    tonics_file: Path

    def tonics(self) -> dict[str, float]:
        with self.tonics_file.open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def performers(self, raga: str) -> list[str]:
        return sorted(path.stem for path in (self.pitch_track_dir / raga).glob("*.tsv"))

    def annotation_path(self, raga: str, performer: str) -> Path:
        return self.annotation_dir / raga / f"{performer}_{raga}.tsv"

    def pitch_track_path(self, raga: str, performer: str) -> Path:
        return self.pitch_track_dir / raga / f"{performer}.tsv"

    def audio_path(self, raga: str, performer: str) -> Path | None:
        candidate = self.audio_dir / raga / f"{performer}.mp3"
        return candidate if candidate.exists() else None


def read_pitch_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    track = pd.read_csv(path, sep="\t", header=None, names=["time", "frequency"])
    return track["time"].to_numpy(dtype=float), track["frequency"].to_numpy(dtype=float)


def read_elan_annotations(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "start": annotations["Begin time"].map(parse_timestamp),
            "end": annotations["End time"].map(parse_timestamp),
            "label": annotations["Annotation"],
        }
    )


def parse_timestamp(timestamp: str) -> float:
    fields = [float(field) for field in str(timestamp).split(":")]
    seconds = 0.0
    for field in fields:
        seconds = seconds * 60 + field
    return seconds


def svara_of(annotation: str) -> str:
    return str(annotation)[0]


def hz_to_cents(frequency: npt.ArrayLike, tonic: float) -> npt.NDArray[np.float64]:
    return 1200.0 * np.log2(np.asarray(frequency, dtype=float) / tonic)


def cents_to_hz(cents: npt.ArrayLike, tonic: float) -> npt.NDArray[np.float64]:
    return tonic * 2.0 ** (np.asarray(cents, dtype=float) / 1200.0)


def normalise_cents(cents: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return (np.asarray(cents, dtype=float) + 4200.0) / 8400.0


def denormalise_cents(normalised: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(normalised, dtype=float) * 8400.0 - 4200.0


def unvoiced_to_nan(frequency: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(frequency, dtype=float).copy()
    values[values == 0] = np.nan
    return values


def interpolate_gaps(
    values: npt.ArrayLike,
    gap_value: float,
    max_gap: float,
    protected_indices: Iterable[int] | None = None,
) -> npt.NDArray[np.float64]:
    contour = np.asarray(values, dtype=float).copy()
    protected = set(protected_indices or ())
    is_gap = np.isnan(contour) if np.isnan(gap_value) else contour == gap_value

    for start, stop in runs(is_gap):
        if stop - start > max_gap:
            continue
        if protected & set(range(start, stop)):
            continue
        contour[start:stop] = np.nan

    return pd.Series(contour).interpolate(method="linear").ffill().bfill().to_numpy()


def smooth_contour(
    times: npt.ArrayLike,
    cents: npt.ArrayLike,
    smoothing_factor: float,
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


def extract_pitch(
    audio_path: Path,
    fmin: float = 32.7,
    fmax: float = 1200.0,
    sample_rate: int = 44100,
    hop_length: int = 441,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    import librosa

    frequency, _ = librosa.pyin(
        str(audio_path),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length,
        fill_na=0.0,
    )
    times = np.arange(len(frequency)) * hop_length / sample_rate
    return times, frequency.astype(float)


def runs(flags: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(flags, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


@dataclass(frozen=True)
class Performance:
    raga: str
    performer: str
    tonic: float
    times: npt.NDArray[np.float64]
    contour: npt.NDArray[np.float64]
    voicing: npt.NDArray[np.float64]
    targets: npt.NDArray[np.int64]
    boundaries: npt.NDArray[np.bool_]

    @property
    def num_frames(self) -> int:
        return len(self.targets)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "raga": self.raga,
                "performer": self.performer,
                "tonic": self.tonic,
                "times": torch.from_numpy(self.times),
                "contour": torch.from_numpy(self.contour),
                "voicing": torch.from_numpy(self.voicing),
                "targets": torch.from_numpy(self.targets),
                "boundaries": torch.from_numpy(self.boundaries),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Performance:
        stored = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            raga=stored["raga"],
            performer=stored["performer"],
            tonic=float(stored["tonic"]),
            times=stored["times"].numpy(),
            contour=stored["contour"].numpy(),
            voicing=stored["voicing"].numpy(),
            targets=stored["targets"].numpy(),
            boundaries=stored["boundaries"].numpy(),
        )


def frame_targets(
    times: npt.NDArray[np.float64],
    annotations: npt.NDArray[np.float64],
    labels: npt.NDArray[np.str_],
    svaras: tuple[str, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_], int]:
    none = len(svaras)
    targets = np.full(len(times), none, dtype=np.int64)
    onsets = np.zeros(len(times), dtype=bool)
    skipped = 0

    for start, end, label in zip(annotations[:, 0], annotations[:, 1], labels, strict=True):
        svara = svara_of(label)
        if svara not in svaras:
            skipped += 1
            continue
        frames = np.where((times >= start) & (times < end))[0]
        if frames.size == 0:
            continue
        targets[frames] = svaras.index(svara)
        onsets[frames[0]] = True

    changes = np.zeros(len(times), dtype=bool)
    changes[1:] = targets[1:] != targets[:-1]
    boundaries = changes | onsets
    boundaries[0] = False
    return targets, boundaries, skipped


def build_performance(
    corpus: VarnamCorpus,
    raga: str,
    performer: str,
    svaras: tuple[str, ...],
    *,
    smoothing_factor: float = 0.5,
    interpolation_gap: float = 0.02,
) -> Performance:
    tonic = float(corpus.tonics()[performer])
    times, frequency = read_pitch_track(corpus.pitch_track_path(raga, performer))

    raw = unvoiced_to_nan(frequency)
    voicing = (~np.isnan(raw)).astype(float)
    filled = interpolate_gaps(raw, np.nan, interpolation_gap)
    filled = smooth_contour(times, filled, smoothing_factor)
    cents = normalise_cents(hz_to_cents(filled, tonic))

    annotations = read_elan_annotations(corpus.annotation_path(raga, performer))
    targets, boundaries, skipped = frame_targets(
        times, annotations[["start", "end"]].to_numpy(dtype=float), annotations["label"].to_numpy(), svaras
    )
    if skipped:
        warn(f"{raga}/{performer}: skipped {skipped} annotations outside the seven svaras")

    return Performance(
        raga=raga,
        performer=performer,
        tonic=tonic,
        times=times,
        contour=cents,
        voicing=voicing,
        targets=targets,
        boundaries=boundaries,
    )


def split_performances(
    performances: list[Performance],
    test_performer: str,
) -> tuple[list[Performance], list[Performance]]:
    train = [p for p in performances if p.performer != test_performer]
    test = [p for p in performances if p.performer == test_performer]
    return train, test


def build_performances(
    corpus: VarnamCorpus,
    ragas: tuple[str, ...],
    svaras: tuple[str, ...],
    frames_dir: Path,
) -> list[Performance]:
    performances = []
    for raga in ragas:
        for performer in corpus.performers(raga):
            cache = frames_dir / raga / f"{performer}.pt"
            if cache.exists():
                performances.append(Performance.load(cache))
                continue
            performance = build_performance(corpus, raga, performer, svaras)
            performance.save(cache)
            performances.append(performance)
    return performances

from __future__ import annotations

from pathlib import Path

import Levenshtein
import mir_eval
import numpy as np
import yaml


def svaras() -> tuple[str, ...]:
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
    )
    return tuple(config["svaras"])


def notes_to_events(
    times: np.ndarray, notes: list[tuple[int, int, int]]
) -> list[tuple[float, float, str]]:
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    return [
        (float(times[start]), float(times[stop - 1]) + step, svaras()[label])
        for start, stop, label in notes
    ]


def reference_events(
    times: np.ndarray,
    begins: np.ndarray,
    finishes: np.ndarray,
    annotations: list[str],
) -> list[tuple[float, float, str]]:
    symbols = svaras()
    return [
        (float(begins[index]), float(finishes[index]), annotations[index][0])
        for index in range(len(annotations))
        if annotations[index][0] in symbols
        and np.any((times >= begins[index]) & (times < finishes[index]))
    ]


def in_coverage(
    events: list[tuple[float, float, str]], coverage: list[tuple[float, float]]
) -> list[tuple[float, float, str]]:
    return [
        event
        for event in events
        if any(event[0] >= begin and event[0] < finish for begin, finish in coverage)
    ]


def lcs_length(first: str, second: str) -> int:
    previous = [0] * (len(second) + 1)
    for character in first:
        current = [0]
        for index, other in enumerate(second):
            current.append(
                previous[index] + 1
                if character == other
                else max(current[-1], previous[index + 1])
            )
        previous = current
    return previous[-1]


def lcs_text(first: str, second: str) -> str:
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for row, character in enumerate(first, start=1):
        for column, other in enumerate(second, start=1):
            table[row][column] = (
                table[row - 1][column - 1] + 1
                if character == other
                else max(table[row - 1][column], table[row][column - 1])
            )
    common: list[str] = []
    row, column = len(first), len(second)
    while row > 0 and column > 0:
        if first[row - 1] == second[column - 1]:
            common.append(first[row - 1])
            row -= 1
            column -= 1
        elif table[row - 1][column] >= table[row][column - 1]:
            row -= 1
        else:
            column -= 1
    return "".join(reversed(common))


def phrase_rows(
    phrases: list[list[float]],
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    window: float,
    count: int = 10,
) -> list[tuple[str, str, float, str, float, float]]:
    symbols = svaras()
    ordered = sorted(phrases, key=lambda span: span[1] - span[0], reverse=True)
    rows: list[tuple[str, str, float, str, float, float]] = []
    for begin, finish in ordered[:count]:
        ref_notes = [note for note in reference if begin <= note[0] < finish]
        est_notes = [note for note in estimated if begin <= note[0] < finish]
        annotation = "".join(note[2] for note in ref_notes)
        transcription = "".join(note[2] for note in est_notes)
        common = lcs_text(annotation, transcription)
        rows.append(
            (
                transcription,
                annotation,
                Levenshtein.distance(annotation, transcription)
                / max(len(annotation), 1),
                common,
                len(common) / max(len(annotation), 1),
                average_overlap_ratio(ref_notes, est_notes, symbols, window),
            )
        )
    return rows


def average_overlap_ratio(
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    symbols: tuple[str, ...],
    window: float,
) -> float:
    if not reference or not estimated:
        return 0.0
    ref_intervals = np.array([[note[0], note[1]] for note in reference], dtype=float)
    est_intervals = np.array([[note[0], note[1]] for note in estimated], dtype=float)
    ref_pitches = np.array(
        [440.0 * 2 ** (symbols.index(note[2]) / 12) for note in reference]
    )
    est_pitches = np.array(
        [440.0 * 2 ** (symbols.index(note[2]) / 12) for note in estimated]
    )
    return float(
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=0.0,
            offset_min_tolerance=window,
        )[3]
    )


def note_metrics(
    reference: list[tuple[float, float, str]],
    estimated: list[tuple[float, float, str]],
    window: float,
) -> dict[str, float]:
    symbols = svaras()
    ref_intervals = np.array(
        [[note[0], note[1]] for note in reference], dtype=float
    ).reshape(-1, 2)
    est_intervals = np.array(
        [[note[0], note[1]] for note in estimated], dtype=float
    ).reshape(-1, 2)
    ref_pitches = np.array(
        [440.0 * 2 ** (symbols.index(note[2]) / 12) for note in reference]
    )
    est_pitches = np.array(
        [440.0 * 2 ** (symbols.index(note[2]) / 12) for note in estimated]
    )
    flat_ref = np.full(len(reference), 440.0)
    flat_est = np.full(len(estimated), 440.0)
    scores = {
        "con": mir_eval.transcription.onset_precision_recall_f1(
            ref_intervals, est_intervals, onset_tolerance=window
        ),
        "conp": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=None,
        ),
        "conoff": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            flat_ref,
            est_intervals,
            flat_est,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=0.0,
            offset_min_tolerance=window,
        ),
        "conpoff": mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=window,
            pitch_tolerance=50.0,
            offset_ratio=0.0,
            offset_min_tolerance=window,
        ),
    }
    metrics: dict[str, float] = {}
    for key, values in scores.items():
        metrics[f"{key}_precision"] = float(values[0])
        metrics[f"{key}_recall"] = float(values[1])
        metrics[f"{key}_f1"] = float(values[2])
    metrics["aor"] = float(scores["conpoff"][3])

    reference_text = "".join(note[2] for note in reference)
    estimated_text = "".join(note[2] for note in estimated)
    distance = Levenshtein.distance(reference_text, estimated_text)
    lcs = lcs_length(reference_text, estimated_text)
    metrics["levenshtein"] = float(distance)
    metrics["levenshtein_norm"] = distance / max(len(reference_text), 1)
    metrics["lcs"] = float(lcs)
    metrics["lcs_norm"] = lcs / max(len(reference_text), 1)
    metrics["num_reference_notes"] = float(len(reference))
    metrics["num_estimated_notes"] = float(len(estimated))
    return metrics

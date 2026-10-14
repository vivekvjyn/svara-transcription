from __future__ import annotations

import random
from typing import Any

import numpy as np

SVARAS: tuple[str, ...] = ("S", "R", "G", "M", "P", "D", "N")
NONE_INDEX = len(SVARAS)


def voiced_regions(
    voicing: np.ndarray, min_frames: int = 20, gap_frames: int = 10
) -> list[tuple[int, int]]:
    mask = voicing > 0.5
    runs: list[list[int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= gap_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(begin, end) for begin, end in merged if end - begin >= min_frames]


def curvature_peaks(contour: np.ndarray, min_gap: int) -> np.ndarray:
    values = np.nan_to_num(contour, nan=0.0)
    dx = np.gradient(values)
    dy = np.gradient(dx)
    ddx = np.gradient(dy)
    ddy = np.gradient(ddx)
    numerator = np.abs(ddx * dy - dx * ddy)
    denominator = np.power(dx * dx + dy * dy, 1.5) + 1e-12
    curvature = numerator / denominator
    peaks: list[int] = []
    for index in range(1, len(curvature) - 1):
        if curvature[index] >= curvature[index - 1] and curvature[index] > curvature[index + 1]:
            if peaks and index - peaks[-1] < min_gap:
                if curvature[index] > curvature[peaks[-1]]:
                    peaks[-1] = index
                continue
            peaks.append(index)
    return np.asarray(peaks, dtype=np.int64)


def initial_boundaries(
    peaks: np.ndarray, start: int, stop: int, dmin: int, dmax: int
) -> np.ndarray:
    boundaries = [start]
    for peak in peaks:
        if peak - boundaries[-1] < dmin:
            continue
        if stop - peak < dmin:
            break
        boundaries.append(int(peak))
    boundaries.append(stop)
    while True:
        gaps = np.diff(boundaries)
        oversized = np.flatnonzero(gaps > dmax)
        if oversized.size == 0:
            break
        index = int(oversized[0])
        mid = boundaries[index] + gaps[index] // 2
        boundaries.insert(index + 1, int(mid))
    return np.asarray(boundaries, dtype=np.int64)


def mutate(
    boundaries: np.ndarray,
    dmin: int,
    dmax: int,
    smax: int,
    pm: float,
    pd: float,
    pa: float,
    rng: random.Random,
) -> np.ndarray:
    values: list[int | None] = [int(value) for value in boundaries]
    for index in range(1, len(values) - 1):
        current = values[index]
        left = values[index - 1]
        right = values[index + 1]
        if left is None or right is None:
            continue
        width = right - left
        if current is None:
            if (rng.random() < pa or width > dmax) and width // 2 >= dmin:
                values[index] = left + width // 2
            continue
        if rng.random() < pd and width <= dmax:
            values[index] = None
            continue
        if rng.random() < pm:
            candidates = [
                value
                for value in range(current - smax, current + smax + 1)
                if value - left >= dmin and right - value >= dmin
            ]
            if candidates:
                values[index] = candidates[rng.randrange(len(candidates))]
    kept = sorted({value for value in values if value is not None})
    return np.asarray(kept, dtype=np.int64)


def crossover(
    first: np.ndarray, second: np.ndarray, pc: float, rng: random.Random
) -> np.ndarray:
    if rng.random() >= pc or len(first) < 3 or len(second) < 3:
        return first.copy()
    cut = rng.randrange(1, len(first) - 1)
    merged = np.unique(np.concatenate([first[: cut + 1], second[cut + 1 :]]))
    if merged[0] != first[0]:
        merged = np.concatenate([[first[0]], merged])
    if merged[-1] != first[-1]:
        merged = np.concatenate([merged, [first[-1]]])
    return merged


def segment_embeddings(
    prefix: np.ndarray, boundaries: np.ndarray, context: int
) -> np.ndarray:
    starts = boundaries[:-1]
    stops = boundaries[1:]
    frames = prefix.shape[1] - 1

    def pool(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lengths = np.maximum(right - left, 1).astype(np.float64)[np.newaxis, :]
        return ((prefix[:, right] - prefix[:, left]) / lengths).T

    prec_left = np.maximum(starts - context, 0)
    succ_right = np.minimum(stops + context, frames)
    current = pool(starts, stops)
    prec = pool(prec_left, starts)
    succ = pool(stops, succ_right)
    return np.concatenate([prec, current, succ], axis=1)


def batch_costs(
    population: list[np.ndarray],
    prefix: np.ndarray,
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    beat_frames: float,
    context: int,
) -> np.ndarray:
    counts = np.asarray([max(len(boundaries) - 1, 1) for boundaries in population])
    embeddings = np.concatenate(
        [segment_embeddings(prefix, boundaries, context) for boundaries in population]
    )
    probabilities = classifier.predict_proba(embeddings)
    labels = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(labels)), labels]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    costs = np.empty(len(population))
    for index in range(len(population)):
        member = population[index]
        segment_labels = labels[offsets[index] : offsets[index + 1]]
        segment_confidence = confidence[offsets[index] : offsets[index + 1]]
        l_svara = float(segment_confidence.mean())
        if len(segment_labels) < 3:
            l_trans = float(unigram[segment_labels].mean())
        else:
            current = segment_labels[2:]
            first = segment_labels[:-2]
            second = segment_labels[1:-1]
            pairs = bigram[first, second]
            triplets = trigram[first, second, current]
            observed = np.where(
                pairs > 0, (triplets + 1e-3) / (pairs + 1e-3 * len(SVARAS)), unigram[current]
            )
            l_trans = float(observed.mean())
        lengths_beats = np.diff(member) / max(beat_frames, 1.0)
        l_length = float(np.minimum(lengths_beats, 8.0).mean() / 8.0)
        costs[index] = alpha * l_svara + beta * l_trans - gamma * l_length
    return costs


def optimize_chunk(
    contour: np.ndarray,
    start: int,
    stop: int,
    prefix: np.ndarray,
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    dmin: int,
    dmax: int,
    smax: int,
    alpha: float,
    beta: float,
    gamma: float,
    beat_frames: float,
    context: int,
    population_size: int,
    iterations: int,
    pm: float,
    pc: float,
    pd: float,
    pa: float,
    rng: random.Random,
) -> np.ndarray:
    peaks = curvature_peaks(contour[start:stop], dmin) + start
    seed = initial_boundaries(peaks, start, stop, dmin, dmax)
    population = [seed]
    while len(population) < population_size:
        mutated = mutate(seed.copy(), dmin, dmax, smax, pm, pd, pa, rng)
        population.append(mutated if len(mutated) >= 3 else seed.copy())
    scores = batch_costs(
        population, prefix, classifier, trigram, bigram, unigram,
        alpha, beta, gamma, beat_frames, context,
    )
    for _ in range(iterations):
        order = np.argsort(scores)[::-1]
        parent_count = max(2, population_size // 4)
        parents = [population[index] for index in order[:parent_count]]
        next_population = [population[order[0]].copy()]
        next_scores = [scores[order[0]]]
        while len(next_population) < population_size:
            first = parents[rng.randrange(len(parents))]
            second = parents[rng.randrange(len(parents))]
            child = mutate(
                crossover(first, second, pc, rng), dmin, dmax, smax, pm, pd, pa, rng
            )
            if len(child) < 3 or child[0] != start or child[-1] != stop:
                child = first.copy()
            next_population.append(child)
            next_scores.append(0.0)
        evolved = batch_costs(
            next_population[1:], prefix, classifier, trigram, bigram, unigram,
            alpha, beta, gamma, beat_frames, context,
        )
        next_scores[1:] = evolved
        population = next_population
        scores = np.asarray(next_scores)
    return population[int(np.argmax(scores))]


def genetic_transcribe(
    contour: np.ndarray,
    frame_embeddings: np.ndarray,
    regions: list[tuple[int, int]],
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    beat_frames: float,
    context: int,
    alpha: float,
    beta: float,
    gamma: float,
    population_size: int,
    iterations: int,
    pm: float,
    pc: float,
    pd: float,
    pa: float,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    frames = len(contour)
    dmin = max(int(round(beat_frames * 0.5)), 2)
    dmax = max(int(round(beat_frames * 2.0)), dmin + 1)
    smax = max(int(round(beat_frames)), 1)
    chunk_frames = max(dmax * 6, dmin * 10)
    prefix = np.zeros((frame_embeddings.shape[1], frames + 1), dtype=np.float64)
    prefix[:, 1:] = np.cumsum(np.asarray(frame_embeddings, dtype=np.float64), axis=0).T
    notes: list[tuple[int, int, int]] = []
    for region_start, region_stop in regions:
        chunk_start = int(region_start)
        while chunk_start < region_stop - dmin:
            chunk_stop = min(chunk_start + chunk_frames, region_stop)
            if region_stop - chunk_stop < dmin:
                chunk_stop = region_stop
            if chunk_stop - chunk_start < 2 * dmin:
                break
            boundaries = optimize_chunk(
                contour,
                chunk_start,
                chunk_stop,
                prefix,
                classifier,
                trigram,
                bigram,
                unigram,
                dmin,
                dmax,
                smax,
                alpha,
                beta,
                gamma,
                beat_frames,
                context,
                population_size,
                iterations,
                pm,
                pc,
                pd,
                pa,
                rng,
            )
            if len(boundaries) >= 2:
                labels = classifier.predict(
                    segment_embeddings(prefix, boundaries, context)
                ).astype(np.int64)
                for index in range(len(labels)):
                    if labels[index] != NONE_INDEX:
                        notes.append(
                            (
                                int(boundaries[index]),
                                int(boundaries[index + 1]),
                                int(labels[index]),
                            )
                        )
            chunk_start = chunk_stop
    return notes

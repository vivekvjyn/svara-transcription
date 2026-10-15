from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np

SVARAS: tuple[str, ...] = ("S", "R", "G", "M", "P", "D", "N")
NONE_INDEX = len(SVARAS)

EmbedFn = Callable[[list[tuple[int, int]]], np.ndarray]


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


def batch_costs(
    population: list[np.ndarray],
    embed_fn: EmbedFn,
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    counts = np.asarray([max(len(boundaries) - 1, 1) for boundaries in population])
    pairs = [
        pair
        for member in population
        for pair in zip(member[:-1].tolist(), member[1:].tolist(), strict=True)
    ]
    embeddings = embed_fn(pairs)
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
            pairs_counts = bigram[first, second]
            triplets = trigram[first, second, current]
            observed = np.where(
                pairs_counts > 0,
                (triplets + 1e-3) / (pairs_counts + 1e-3 * len(SVARAS)),
                unigram[current],
            )
            l_trans = float(observed.mean())
        costs[index] = (alpha * l_svara + beta * l_trans) / (alpha + beta)
    return costs


def optimize_chunk(
    contour: np.ndarray,
    start: int,
    stop: int,
    embed_fn: EmbedFn,
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    dmin: int,
    dmax: int,
    smax: int,
    alpha: float,
    beta: float,
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
        population, embed_fn, classifier, trigram, bigram, unigram,
        alpha, beta,
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
            next_population[1:], embed_fn, classifier, trigram, bigram, unigram,
            alpha, beta,
        )
        next_scores[1:] = evolved
        population = next_population
        scores = np.asarray(next_scores)
    return population[int(np.argmax(scores))]


def genetic_transcribe(
    contour: np.ndarray,
    regions: list[tuple[int, int]],
    embed_fn: EmbedFn,
    classifier: Any,
    trigram: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    beat_frames: float,
    alpha: float,
    beta: float,
    population_size: int,
    iterations: int,
    pm: float,
    pc: float,
    pd: float,
    pa: float,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    dmin = max(int(round(beat_frames * 0.5)), 2)
    dmax = max(int(round(beat_frames * 2.0)), dmin + 1)
    smax = max(int(round(beat_frames)), 1)
    chunk_frames = max(dmax * 6, dmin * 10)
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
                embed_fn,
                classifier,
                trigram,
                bigram,
                unigram,
                dmin,
                dmax,
                smax,
                alpha,
                beta,
                population_size,
                iterations,
                pm,
                pc,
                pd,
                pa,
                rng,
            )
            if len(boundaries) >= 2:
                pairs = list(
                    zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True)
                )
                labels = classifier.predict(embed_fn(pairs)).astype(np.int64)
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

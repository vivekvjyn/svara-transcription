from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np


class GeneticOptimiser:
    """Genetic optimiser for svara boundary sequences.

    Holds the trained classifier, svara transition model, genetic
    parameters, and random state shared by every operation of one run.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[tuple[int, int]]], np.ndarray],
        classifier: Any,
        trigram: np.ndarray,
        bigram: np.ndarray,
        unigram: np.ndarray,
        svaras: tuple[str, ...],
        args: Any,
        rng: random.Random,
    ) -> None:
        self.embed_fn = embed_fn
        self.classifier = classifier
        self.trigram = trigram
        self.bigram = bigram
        self.unigram = unigram
        self.svaras = svaras
        self.population_size = args.ga_population
        self.iterations = args.ga_iterations
        self.pm = args.pm
        self.pc = args.pc
        self.pd = args.pd
        self.pa = args.pa
        self.rng = rng

    @staticmethod
    def beat_lattice(beats: np.ndarray) -> np.ndarray:
        if beats.size == 0:
            return np.asarray([], dtype=np.int64)
        if beats.size == 1:
            return np.asarray([int(beats[0])], dtype=np.int64)
        points: list[int] = []
        for first, second in zip(beats[:-1], beats[1:], strict=True):
            span = int(second) - int(first)
            points.extend(int(first) + round(step * span / 4) for step in range(4))
        points.append(int(beats[-1]))
        return np.asarray(sorted(set(points)), dtype=np.int64)

    @staticmethod
    def _snap(lattice: np.ndarray, position: int, low: int, high: int) -> int | None:
        if lattice.size == 0 or high < low:
            return None
        index = int(np.searchsorted(lattice, position))
        best: tuple[float, int] | None = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < lattice.size:
                value = int(lattice[candidate])
                if low <= value <= high:
                    distance = abs(value - position)
                    if best is None or distance < best[0]:
                        best = (distance, value)
        return best[1] if best else None

    @staticmethod
    def initial_boundaries(
        peaks: np.ndarray, start: int, stop: int, dmin: int
    ) -> np.ndarray:
        boundaries = [start]
        for peak in peaks:
            if peak - boundaries[-1] < dmin:
                continue
            if stop - peak < dmin:
                break
            boundaries.append(int(peak))
        boundaries.append(stop)
        return np.asarray(boundaries, dtype=np.int64)

    @staticmethod
    def repair(
        boundaries: np.ndarray, lattice: np.ndarray, dmax: int
    ) -> np.ndarray:
        values = [int(value) for value in boundaries]
        index = 0
        while index < len(values) - 1:
            gap = values[index + 1] - values[index]
            if gap > dmax:
                snapped = GeneticOptimiser._snap(
                    lattice,
                    values[index] + gap // 2,
                    values[index] + 1,
                    values[index + 1] - 1,
                )
                values.insert(
                    index + 1, snapped if snapped is not None else values[index] + gap // 2
                )
                if values[index + 1] - values[index] > dmax:
                    continue
            index += 1
        return np.asarray(values, dtype=np.int64)

    def mutate(
        self,
        boundaries: np.ndarray,
        lattice: np.ndarray,
        dmin: int,
        dmax: int,
        smax: int,
    ) -> np.ndarray:
        rng = self.rng
        values: list[int | None] = [int(value) for value in boundaries]
        for index in range(1, len(values) - 1):
            current = values[index]
            left = values[index - 1]
            right = values[index + 1]
            if left is None or right is None:
                continue
            width = right - left
            if rng.random() < self.pd and width <= dmax:
                values[index] = None
                continue
            if rng.random() < self.pm:
                low, high = current - smax, current + smax
                index_low = int(np.searchsorted(lattice, low))
                index_high = int(np.searchsorted(lattice, high + 1))
                candidates = [
                    int(value)
                    for value in lattice[index_low:index_high]
                    if value - left >= dmin and right - value >= dmin
                ]
                if candidates:
                    values[index] = candidates[rng.randrange(len(candidates))]
        for index in range(1, len(values) - 1):
            if values[index] is not None:
                continue
            left = values[index - 1]
            right = values[index + 1]
            if left is None or right is None:
                continue
            width = right - left
            if (rng.random() < self.pa or width > dmax) and width // 2 >= dmin:
                snapped = self._snap(
                    lattice, left + width // 2, left + dmin, right - dmin
                )
                if snapped is not None:
                    values[index] = snapped
        kept = sorted({value for value in values if value is not None})
        return np.asarray(kept, dtype=np.int64)

    def crossover(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() >= self.pc or len(first) < 3 or len(second) < 3:
            return first.copy()
        cut = rng.randrange(1, len(first) - 1)
        merged = np.unique(np.concatenate([first[: cut + 1], second[cut + 1 :]]))
        if merged[0] != first[0]:
            merged = np.concatenate([[first[0]], merged])
        if merged[-1] != first[-1]:
            merged = np.concatenate([merged, [first[-1]]])
        return merged

    def batch_costs(
        self,
        population: list[np.ndarray],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        beat_frames: float,
        voiced: np.ndarray,
    ) -> np.ndarray:
        counts = np.asarray([max(len(boundaries) - 1, 1) for boundaries in population])
        pairs = [
            pair
            for member in population
            for pair in zip(member[:-1].tolist(), member[1:].tolist(), strict=True)
        ]
        embeddings = self.embed_fn(pairs)
        probabilities = self.classifier.predict_proba(embeddings)
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
                l_trans = float(self.unigram[segment_labels].mean())
            else:
                current = segment_labels[2:]
                first = segment_labels[:-2]
                second = segment_labels[1:-1]
                pairs_counts = self.bigram[first, second]
                triplets = self.trigram[first, second, current]
                observed = np.where(
                    pairs_counts > 0,
                    (triplets + 1e-3) / (pairs_counts + 1e-3 * len(self.svaras)),
                    self.unigram[current],
                )
                l_trans = float(observed.mean())
            lengths_beats = np.diff(member) / max(beat_frames, 1.0)
            l_length = float(np.exp(-lengths_beats).mean())
            none_index = len(self.svaras)
            l_none = 0.0
            for frame_start, frame_stop, is_none in zip(
                member[:-1], member[1:], segment_labels == none_index, strict=True
            ):
                if is_none and voiced[frame_start:frame_stop].any():
                    l_none += 2.0 * float(frame_stop - frame_start) / max(
                        beat_frames, 1.0
                    )
            costs[index] = (
                (alpha * l_svara + beta * l_trans) / (alpha + beta)
                + gamma * l_length
                - delta * l_none
            )
        return costs

    def optimise_chunk(
        self,
        contour: np.ndarray,
        start: int,
        stop: int,
        lattice: np.ndarray,
        dmin: int,
        dmax: int,
        smax: int,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        beat_frames: float,
    ) -> np.ndarray:
        rng = self.rng
        voiced = np.isfinite(contour)
        low = int(np.searchsorted(lattice, start))
        high = int(np.searchsorted(lattice, stop))
        candidates = lattice[low:high]
        seed = self.repair(
            self.initial_boundaries(candidates, start, stop, dmin), lattice, dmax
        )
        population = [seed]
        while len(population) < self.population_size:
            mutated = self.mutate(seed.copy(), lattice, dmin, dmax, smax)
            population.append(mutated if len(mutated) >= 3 else seed.copy())
        scores = self.batch_costs(
            population, alpha, beta, gamma, delta, beat_frames, voiced
        )
        for _ in range(self.iterations):
            order = np.argsort(scores)[::-1]
            parent_count = max(2, self.population_size // 4)
            parents = [population[index] for index in order[:parent_count]]
            next_population = [population[order[0]].copy()]
            next_scores = [scores[order[0]]]
            while len(next_population) < self.population_size:
                first = parents[rng.randrange(len(parents))]
                second = parents[rng.randrange(len(parents))]
                child = self.repair(
                    self.mutate(self.crossover(first, second), lattice, dmin, dmax, smax),
                    lattice,
                    dmax,
                )
                if len(child) < 3 or child[0] != start or child[-1] != stop:
                    child = first.copy()
                next_population.append(child)
                next_scores.append(0.0)
            evolved = self.batch_costs(
                next_population[1:], alpha, beta, gamma, delta, beat_frames, voiced
            )
            next_scores[1:] = evolved
            population = next_population
            scores = np.asarray(next_scores)
        return population[int(np.argmax(scores))]

    def transcribe(
        self,
        contour: np.ndarray,
        regions: list[tuple[int, int]],
        beats: np.ndarray,
        beat_frames: float,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
    ) -> list[tuple[int, int, int]]:
        dmin = max(int(round(beat_frames * 0.25)), 2)
        dmax = max(int(round(beat_frames * 4.0)), dmin + 1)
        smax = max(int(round(beat_frames)), 1)
        chunk_frames = max(dmax * 6, dmin * 10)
        lattice = self.beat_lattice(beats)
        notes: list[tuple[int, int, int]] = []
        for region_start, region_stop in regions:
            chunk_start = int(region_start)
            while chunk_start < region_stop - dmin:
                chunk_stop = min(chunk_start + chunk_frames, region_stop)
                if region_stop - chunk_stop < dmin:
                    chunk_stop = region_stop
                if chunk_stop - chunk_start < 2 * dmin:
                    break
                boundaries = self.optimise_chunk(
                    contour,
                    chunk_start,
                    chunk_stop,
                    lattice,
                    dmin,
                    dmax,
                    smax,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    beat_frames,
                )
                if len(boundaries) >= 2:
                    pairs = list(
                        zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True)
                    )
                    labels = self.classifier.predict(self.embed_fn(pairs)).astype(
                        np.int64
                    )
                    for index in range(len(labels)):
                        if labels[index] != len(self.svaras):
                            notes.append(
                                (
                                    int(boundaries[index]),
                                    int(boundaries[index + 1]),
                                    int(labels[index]),
                                )
                            )
                chunk_start = chunk_stop
        return notes

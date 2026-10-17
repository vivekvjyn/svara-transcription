"""Genetic optimisation of svara note boundaries.

Evolves candidate boundary sequences for the annotated regions of a
performance and scores them with the combined svara, transition, and
length fitness of the paper's genetic algorithm.
"""

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
        """Initialise the optimiser.

        :param embed_fn: function mapping boundary pairs to contextual embeddings
        :type embed_fn: Callable[[list[tuple[int, int]]], numpy.ndarray]
        :param classifier: trained svara classifier exposing predict_proba and predict
        :type classifier: Any
        :param trigram: second-order svara transition counts
        :type trigram: numpy.ndarray
        :param bigram: first-order svara transition counts
        :type bigram: numpy.ndarray
        :param unigram: normalised unigram svara distribution
        :type unigram: numpy.ndarray
        :param svaras: svara alphabet; its length is the none-label index
        :type svaras: tuple[str, ...]
        :param args: run arguments (ga_population, ga_iterations, pm, pc, pd, pa)
        :type args: argparse.Namespace
        :param rng: random number generator for reproducible evolution
        :type rng: random.Random
        """
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
        """Build the candidate boundary lattice from the beat track.

        Each inter-beat interval is subdivided into four equal steps, so
        any note length that is a multiple of a 16th note, from a 16th
        up to a whole note, is the distance between two lattice points.

        :param beats: frame indices of the tracked beats
        :type beats: numpy.ndarray
        :return: sorted frame indices of all candidate boundaries
        :rtype: numpy.ndarray
        """
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
        """Nearest lattice point to a position within ``[low, high]``.

        :param lattice: sorted candidate frame indices
        :type lattice: numpy.ndarray
        :param position: frame index to snap to the lattice
        :type position: int
        :param low: lowest acceptable frame index
        :type low: int
        :param high: highest acceptable frame index
        :type high: int
        :return: snapped frame index, or None when no lattice point fits
        :rtype: int | None
        """
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
        """Build a seed boundary sequence covering a chunk.

        :param peaks: candidate boundary frame indices (beat lattice)
        :type peaks: numpy.ndarray
        :param start: chunk start frame
        :type start: int
        :param stop: chunk end frame
        :type stop: int
        :param dmin: minimum distance between boundaries in frames
        :type dmin: int
        :return: boundary frame indices from start to stop
        :rtype: numpy.ndarray
        """
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
        """Split any gap longer than dmax until every gap fits.

        Each oversized gap is halved at the lattice point nearest the
        midpoint; a half that is still longer than dmax is halved again,
        so the result never contains a gap above dmax.

        :param boundaries: boundary vector possibly containing oversized gaps
        :type boundaries: numpy.ndarray
        :param lattice: sorted candidate frame indices from the beat track
        :type lattice: numpy.ndarray
        :param dmax: maximum distance between boundaries in frames
        :type dmax: int
        :return: boundary vector with every gap at most dmax
        :rtype: numpy.ndarray
        """
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
        """Apply insert, delete, and shift mutations within dmin/dmax limits.

        Deleted boundaries may be re-inserted at the gap midpoint with
        probability ``pa`` (always when the gap exceeds dmax), so long
        notes split back into shorter ones. Inserted and shifted
        boundaries snap to the beat lattice, so every child position is
        a 16th-note, 8th-note, dotted 8th, or beat position of the
        tracked beat grid.

        :param boundaries: parent boundary vector
        :type boundaries: numpy.ndarray
        :param lattice: sorted candidate frame indices from the beat track
        :type lattice: numpy.ndarray
        :param dmin: minimum distance between boundaries in frames
        :type dmin: int
        :param dmax: maximum distance between boundaries in frames
        :type dmax: int
        :param smax: maximum shift of a boundary in frames
        :type smax: int
        :return: mutated boundary vector
        :rtype: numpy.ndarray
        """
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
        """Combine two parents at a random cut point.

        The offspring keeps the start and stop boundaries of the first
        parent; with probability ``1 - pc`` it is a copy of that parent.

        :param first: first parent boundary vector
        :type first: numpy.ndarray
        :param second: second parent boundary vector
        :type second: numpy.ndarray
        :return: offspring boundary vector
        :rtype: numpy.ndarray
        """
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
        """Evaluate the fitness of every member of a population.

        Each member is labelled with the classifier; the mean label
        confidence (svara), the mean transition probability (trans), and
        the mean exponential decay of the note lengths in beats
        (length, so shorter notes score higher) are combined as
        ``(alpha * svara + beta * trans) / (alpha + beta) + gamma * length
        - delta * none``.  The none term counts, in eighth notes, the
        segments labelled none that overlap voiced pitch frames, so
        vocal material the classifier refuses to name costs one unit
        per eighth note of its length.

        :param population: candidate boundary vectors
        :type population: list[numpy.ndarray]
        :param alpha: weight of the svara term
        :type alpha: float
        :param beta: weight of the transition term
        :type beta: float
        :param gamma: weight of the length term
        :type gamma: float
        :param delta: weight of the none-over-voiced term, per eighth note
        :type delta: float
        :param beat_frames: frame duration of one beat
        :type beat_frames: float
        :param voiced: mask of contour frames carrying a pitch value
        :type voiced: numpy.ndarray
        :return: fitness score of each member (higher is better)
        :rtype: numpy.ndarray
        """
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
        """Evolve the fittest boundary sequence for one chunk.

        Seeds the population from the beat-lattice candidates inside the
        chunk, then applies selection, crossover, and mutation for
        ``args.ga_iterations`` generations.

        :param contour: pitch contour frames of the whole performance
        :type contour: numpy.ndarray
        :param start: chunk start frame
        :type start: int
        :param stop: chunk end frame
        :type stop: int
        :param lattice: sorted candidate frame indices from the beat track
        :type lattice: numpy.ndarray
        :param dmin: minimum distance between boundaries in frames
        :type dmin: int
        :param dmax: maximum distance between boundaries in frames
        :type dmax: int
        :param smax: maximum shift of a boundary in frames
        :type smax: int
        :param alpha: weight of the svara term
        :type alpha: float
        :param beta: weight of the transition term
        :type beta: float
        :param gamma: weight of the length term
        :type gamma: float
        :param delta: weight of the none-over-voiced term, per eighth note
        :type delta: float
        :param beat_frames: frame duration of one beat
        :type beat_frames: float
        :return: best boundary vector found for the chunk
        :rtype: numpy.ndarray
        """
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
        """Transcribe all annotated regions of a performance.

        Splits every region into chunks of ``max(dmax * 6, dmin * 10)``
        frames, optimises each chunk, labels the resulting segments, and
        drops segments classified as none. All candidate boundaries come
        from the beat track: each inter-beat interval is divided into
        16th-note steps; every note length that is a multiple of a 16th,
        from a 16th up to a whole note, is available, with a 16th as the
        minimum and a whole note as the maximum.

        :param contour: pitch contour frames of the whole performance
        :type contour: numpy.ndarray
        :param regions: annotated frame regions to transcribe
        :type regions: list[tuple[int, int]]
        :param beats: frame indices of the tracked beats
        :type beats: numpy.ndarray
        :param beat_frames: frame duration of one beat
        :type beat_frames: float
        :param alpha: weight of the svara term
        :type alpha: float
        :param beta: weight of the transition term
        :type beta: float
        :param gamma: weight of the length term
        :type gamma: float
        :param delta: weight of the none-over-voiced term, per eighth note
        :type delta: float
        :return: notes as (start frame, stop frame, svara label) triples
        :rtype: list[tuple[int, int, int]]
        """
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

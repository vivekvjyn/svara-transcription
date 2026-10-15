import numpy as np
from compiam.utils.dbn import HiddenMarkovModel


def dbn_viterbi(self, log_densities):
    num_frames = len(log_densities)
    if num_frames == 0:
        return np.empty(0, dtype=np.int64), -np.inf
    frames = np.asarray(log_densities, dtype=np.float64)
    pointers = self.observation_pointers
    previous = np.full(self.num_states, -np.log(self.num_states))
    current = np.empty(self.num_states)
    backtrack = np.empty((num_frames, len(self._multi)), dtype=np.int32)
    has_multi = len(self._multi) > 0
    for frame in range(num_frames):
        if self._shift:
            current[1:] = previous[:-1]
        else:
            current[self._single] = (
                previous[self._single_prev] + self._single_log_prob
            )
        if has_multi:
            candidates = previous[self._multi_prev] + self._multi_log_prob
            best = candidates.argmax(axis=1)
            current[self._multi] = candidates[self._multi_rows, best]
            backtrack[frame] = self._multi_prev[self._multi_rows, best]
        current += frames[frame, pointers]
        previous, current = current, previous
    state = int(np.argmax(previous))
    log_probability = previous[state]
    if np.isinf(log_probability):
        return np.empty(0, dtype=np.int64), log_probability
    single_lookup = np.zeros(self.num_states, dtype=np.int64)
    single_lookup[self._single] = self._single_prev
    path = np.empty(num_frames, dtype=np.int64)
    for frame in range(num_frames - 1, -1, -1):
        path[frame] = state
        multi = self._multi_index[state]
        state = backtrack[frame, multi] if multi >= 0 else single_lookup[state]
    return path, log_probability


HiddenMarkovModel.viterbi = dbn_viterbi

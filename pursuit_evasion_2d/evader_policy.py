"""Evader policy: run away from pursuer."""
import numpy as np

NUM_ACCEL_ACTIONS = 9


def scripted_evader_run_away(evader_pos, pursuer_pos, evader_a_max, np_random=None):
    if np_random is None:
        np_random = np.random.default_rng()
    diff = evader_pos - pursuer_pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np_random.integers(1, NUM_ACCEL_ACTIONS)
    direction = diff / dist
    angle = np.arctan2(direction[1], direction[0])
    if angle < 0:
        angle += 2 * np.pi
    idx = int(round(angle / (2 * np.pi) * 8)) % 8
    return idx + 1

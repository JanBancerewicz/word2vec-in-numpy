import json
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


def unigram_noise(freqs: np.ndarray) -> np.ndarray:
    smoothed = freqs ** 0.75  # 0.75 power boosts rare words relative to their raw frequency
    return smoothed / smoothed.sum()

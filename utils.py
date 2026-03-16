import json 
from pathlib import Path
from typing import Any
import numpy as np

def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

def unigram_noise(freqs: np.ndarray) -> np.ndarray:
    smoothed = freqs ** 0.75
    return smoothed / smoothed.sum()

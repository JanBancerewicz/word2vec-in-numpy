from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_corpus(filepath: str, max_tokens: Optional[int] = None) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        tokens = f.read().lower().split()
    return tokens[:max_tokens] if max_tokens else tokens  # cap at max_tokens if set


def build_vocab(
    tokens: List[str], min_count: int = 5
) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    word_counts    = Counter(tokens)
    filtered_words = sorted(
        w for w, c in word_counts.items()
        if c >= min_count and len(w) >= 2  # drop rare words and single-char tokens
    )
    word2id  = {w: i for i, w in enumerate(filtered_words)}
    id2word  = {i: w for i, w in enumerate(filtered_words)}
    freq_arr = np.array([word_counts[w] for w in filtered_words], dtype=np.float64)
    return word2id, id2word, freq_arr


def subsample(
    corpus_ids: np.ndarray, freq_arr: np.ndarray, threshold: float = 1e-5
) -> np.ndarray:
    corpus_size  = freq_arr.sum()
    discard_prob = 1.0 - np.sqrt(threshold * corpus_size / freq_arr)  # more frequent → higher drop probability
    discard_prob = np.clip(discard_prob, 0, 1)
    retained     = np.random.random(len(corpus_ids)) >= discard_prob[corpus_ids]
    return corpus_ids[retained]


def prepare_corpus(
    corpus_path: str,
    max_tokens: Optional[int],
    min_count: int,
    threshold: float = 1e-5,
):
    tokens     = load_corpus(corpus_path, max_tokens=max_tokens)
    word2id, id2word, freq_arr = build_vocab(tokens, min_count=min_count)
    corpus_ids = np.array([word2id[w] for w in tokens if w in word2id])  # OOV words skipped
    corpus_ids = subsample(corpus_ids, freq_arr, threshold=threshold)
    return tokens, corpus_ids, word2id, id2word, freq_arr

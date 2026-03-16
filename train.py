import numpy as np
from skipgram_model import SkipGramModel


def train(
    corpus_ids: np.ndarray,
    vocab_size: int,
    noise_dist: np.ndarray,
    embed_dim: int = 100,
    window: int = 7,
    n_neg: int = 5,
    lr: float = 0.04,
    epochs: int = 2,
) -> np.ndarray:
    model = SkipGramModel(vocab_size, embed_dim)
    total_steps = epochs * len(corpus_ids)  # used for lr decay schedule

    for epoch in range(1, epochs + 1):
        model.fit_epoch(corpus_ids, noise_dist, window, n_neg, lr, epoch, epochs, total_steps)

    return model.embeddings


def nearest_words(word, W, word2id, id2word, top_n: int = 10) -> list[tuple[str, float]]:
    if word not in word2id:
        print(f"'{word}' not in vocabulary")
        return []

    idx   = word2id[word]
    query = W[idx] / np.linalg.norm(W[idx]) # unit vector for cosine similarity

    norms        = np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)
    similarities = (W / norms) @ query # cosine similarity against all words
    similarities[idx] = -np.inf # exclude the query word itself

    top_indices = np.argpartition(similarities, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = [(id2word[i], float(similarities[i])) for i in top_indices]
    print(f"\n  similar to '{word}':")
    print(f"  {'word':<15} similarity")
    print(f"  {'-'*26}")
    for w, score in results:
        bar = "█" * int(score * 20)
        print(f"  {w:<15} {score:.4f}  {bar}")
    return results
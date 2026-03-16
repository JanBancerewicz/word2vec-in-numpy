import numpy as np


def load_embeddings(model_path: str) -> tuple[np.ndarray, dict, dict]:
    data    = np.load(model_path, allow_pickle=True)
    W       = data["W"]
    words   = data["words"].tolist()
    word2id = {w: i for i, w in enumerate(words)}  # word -> row index in W
    id2word = {i: w for i, w in enumerate(words)}   # row index -> word
    print(f"loaded {len(words)} words, embed_dim={W.shape[1]}")
    return W, word2id, id2word

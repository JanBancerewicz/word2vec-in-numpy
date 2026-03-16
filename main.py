from pathlib import Path

import numpy as np

from src.load_embeddings import load_embeddings
from src.preprocess import prepare_corpus
from src.train import train, nearest_words
from src.utils import unigram_noise, _json_load

ROOT       = Path(__file__).parent
DATA_PATH  = ROOT / "data" / "text8"
MODEL_PATH = ROOT / "data" / "saved_model.npz"
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config not found: {CONFIG_PATH}")
    return _json_load(CONFIG_PATH)


def run_training(cfg: dict) -> tuple:
    if not DATA_PATH.exists():
        print("Run download_data.py first.")
        raise SystemExit(1)

    print("preprocessing ...")
    tokens, corpus_ids, word2id, id2word, freq_arr = prepare_corpus(
        str(DATA_PATH),
        max_tokens=cfg["max_tokens"],
        min_count=cfg["min_count"],
    )
    vocab_size = len(word2id)
    print(f"{len(tokens)} tokens loaded, {vocab_size} words in vocab")
    print(f"{len(corpus_ids)} tokens after subsampling")

    noise_dist = unigram_noise(freq_arr)  # smoothed unigram for negative sampling
    embeddings = train(
        corpus_ids,
        vocab_size,
        noise_dist,
        embed_dim=cfg["embed_dim"],
        window=cfg["window"],
        n_neg=cfg["n_neg"],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
    )

    np.savez(MODEL_PATH, W=embeddings, words=np.array(list(id2word.values())))
    print(f"\nsaved to {MODEL_PATH}")

    return embeddings, word2id, id2word


if __name__ == "__main__":
    cfg = load_config()

    if MODEL_PATH.exists():
        # skip training — load previously saved embeddings
        print(f"model found at {MODEL_PATH}, loading ...")
        embeddings, word2id, id2word = load_embeddings(str(MODEL_PATH))
    else:
        embeddings, word2id, id2word = run_training(cfg)

    print("\n" + "="*40)
    print("  word similarity search")
    print("  type 'quit' to exit")
    print("="*40)
    while True:
        word = input("\n  query > ").strip().lower()
        if word == "quit":
            break
        nearest_words(word, embeddings, word2id, id2word)

import os

import numpy as np

from load_embeddings import load_embeddings
from preprocess import prepare_corpus
from train import train, nearest_words
from utils import unigram_noise

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "text8")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "saved_model.npz")


if __name__ == "__main__":

    if os.path.exists(MODEL_PATH):
        print(f"model found at {MODEL_PATH}, loading ...")
        W, word2id, id2word = load_embeddings(MODEL_PATH)
    else:
        if not os.path.exists(DATA_PATH):
            print("Run download_data.py first.")
            raise SystemExit(1)

        MAX_TOKENS = 300_000
        MIN_COUNT  = 2

        print("preprocessing ...")
        tokens, corpus_ids, word2id, id2word, freq_arr = prepare_corpus(
            DATA_PATH, max_tokens=MAX_TOKENS, min_count=MIN_COUNT
        )
        V = len(word2id)
        print(f"{len(tokens)} tokens loaded, {V} words in vocab")
        print(f"{len(corpus_ids)} tokens after subsampling")

        noise_dist = unigram_noise(freq_arr)
        W = train(corpus_ids, V, noise_dist, embed_dim=300, window=5, n_neg=5, lr=0.025, epochs=5)

        np.savez(MODEL_PATH, W=W, words=np.array(list(id2word.values())))
        print(f"\nsaved to {MODEL_PATH}")

    print("\ntype a word to find similar ones (or 'quit' to exit)")
    while True:
        word = input("> ").strip().lower()
        if word == "quit":
            break
        nearest_words(word, W, word2id, id2word)

# word2vec-in-numpy

A from-scratch implementation of **Word2Vec skip-gram with negative sampling (SGNS)** using only NumPy.
No PyTorch, no TensorFlow — just matrix math and SGD.

---

## How it works

The model learns dense word vectors by solving a binary classification task: given a center word and a candidate word, predict whether the candidate is a true context neighbour or a noise word drawn from the unigram distribution.

**Objective (per training pair):**

$$\mathcal{L} = -\log \sigma(v_c \cdot u_o) - \sum_{k=1}^{K} \log \sigma(-v_c \cdot u_k)$$

where `v_c` is the center embedding, `u_o` is the positive context vector, and `u_1..u_K` are noise vectors sampled from the smoothed unigram distribution P(w)^0.75.

**Key design choices:**

| Choice | Detail |
|---|---|
| Architecture | Skip-gram |
| Negative sampling | K=5 noise words per pair, unigram^0.75 distribution |
| Window | Dynamic — size drawn from [1, C] per center word |
| Subsampling | Frequent words dropped with P = 1 - sqrt(t / f(w)), t = 1e-5 |
| Learning rate | Linear decay: lr_t = lr_0 * max(0.0001, 1 - t/T) |
| Embeddings | `W_in` matrix (input embeddings) used for similarity queries |

---

## Project structure

```
word2vec-in-numpy/
├── main.py              # entry point: train or load, then interactive query
├── download_data.py     # downloads the text8 corpus
├── config.json          # hyperparameters
├── src/
│   ├── skipgram_model.py    # SkipGramModel — forward pass, gradients, training loop
│   ├── train.py             # train() orchestrator + nearest_words() query
│   ├── preprocess.py        # tokenisation, vocab, subsampling
│   ├── load_embeddings.py   # load saved .npz model
│   └── utils.py             # sigmoid, unigram noise distribution, JSON helpers
└── data/                # corpus and saved model (git-ignored)
```

---

## Quick start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the corpus**

```bash
python download_data.py
```

This downloads [text8](http://mattmahoney.net/dc/text8.zip) (~100 MB cleaned Wikipedia dump) into `data/text8`.

**3. Train**

```bash
python main.py
```

Training reads `config.json`, preprocesses the corpus, runs SGNS for the configured number of epochs, and saves the model to `data/saved_model.npz`.
On subsequent runs the saved model is loaded directly — no retraining.

**4. Query**

After training the program enters an interactive similarity loop:

```
========================================
  word similarity search
  type 'quit' to exit
========================================

  query > war
```
similar to 'war':
  word            similarity
  --------------------------
  this            0.4209  ████████
  including       0.4191  ████████
  philosophy      0.4169  ████████
  an              0.4148  ████████
  virtue          0.4119  ████████
  army            0.4118  ████████
  so              0.4069  ████████
  have            0.4054  ████████
  argentina       0.3996  ███████
  and             0.3991  ███████
---

## Configuration

Edit `config.json` to change hyperparameters:

```json
{
  "max_tokens": 300000,
  "min_count": 2,
  "embed_dim": 300,
  "window": 5,
  "n_neg": 5,
  "lr": 0.025,
  "epochs": 2
}
```

| Key | Description |
|---|---|
| `max_tokens` | How many tokens to read from the corpus |
| `min_count` | Minimum frequency for a word to enter the vocabulary |
| `embed_dim` | Embedding dimensionality |
| `window` | Maximum context window radius |
| `n_neg` | Negative samples per training pair |
| `lr` | Initial learning rate |
| `epochs` | Training epochs |

---

## Requirements

- Python >= 3.11
- NumPy >= 2.0

```bash
pip install numpy
```

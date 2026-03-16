# word2vec-in-numpy

A from-scratch implementation of **Word2Vec skip-gram with negative sampling (SGNS)** using only NumPy.
No PyTorch, no TensorFlow — just matrix math and SGD.

---

## Requirements

- Python >= 3.11
- NumPy >= 2.0

```bash
pip install numpy
```
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
├── tests/
│   └── test_word2vec.py     # unit tests (pytest)
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

  similar to 'war':
  word            similarity
  --------------------------
  alexander       0.5243  ██████████
  appointed       0.5176  ██████████
  authorized      0.5105  ██████████
  born            0.5043  ██████████
  claim           0.4955  █████████
  yielding        0.4933  █████████
  monica          0.4925  █████████
  april           0.4906  █████████
  confederacy     0.4905  █████████
  darius          0.4880  █████████
```

---

## Configuration

Edit `config.json` to change hyperparameters:

```json
{
  "max_tokens": 3000000,
  "min_count": 2,
  "embed_dim": 300,
  "window": 7,
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

## Math behind the implementation

### Objective

For each center word $`w_c`$ and a context word $`w_o`$ drawn from a window of radius $`r`$, the model draws $K$ noise words from the smoothed unigram distribution and optimises:

```math
\mathcal{L} = -\log \sigma(u_o \cdot v_c) - \sum_{i=1}^{K} \log \sigma(-u_{n_i} \cdot v_c)
```

- $`v_c \in \mathbb{R}^d`$ — row of `embeddings` (W_in) for the center word
- $`u_o, u_{n_i} \in \mathbb{R}^d`$ — rows of `context_weights` (W_out) for the positive and noise words

The first term pulls $`v_c`$ and $`u_o`$ together; the negative terms push $`v_c`$ away from noise words.

**Numerical stability.** Rather than computing $`\log \sigma(\cdot)`$ directly, the implementation clips the dot product to $`[-20, 20]`$ before applying sigmoid. This avoids overflow in `exp` without resorting to the `logaddexp` / softplus reformulation.

---

### Gradients

Define the scalar errors:

```math
e_{pos} = \sigma(u_o \cdot v_c) - 1 \qquad e_i = \sigma(u_{n_i}\cdot v_c)
```

Gradient for each output embedding:

```math
\frac{\partial \mathcal{L}}{\partial u_o} = e_{pos}\, v_c
\qquad
\frac{\partial \mathcal{L}}{\partial u_{n_i}} = e_i\, v_c
```

Accumulated gradient for the center embedding across all context words in the window:

```math
\frac{\partial \mathcal{L}}{\partial v_c}
= e_{pos}\, u_o + \sum_{i=1}^{K} e_i\, u_{n_i}
```

The negative-sample sum is vectorised: with $`U_{neg} \in \mathbb{R}^{K \times d}`$ and $`e_{neg} \in \mathbb{R}^K`$,

```math
\sum_{i=1}^{K} e_i\, u_{n_i} = e_{neg} \cdot U_{neg}
```

implemented as a single `(sigma_neg[:, None] * u_neg).sum(axis=0)`.

---

### Parameter updates

`context_weights` (W_out) are updated immediately after each `(center, context)` pair:

```math
u_o \leftarrow u_o - \eta\, e_{pos}\, v_c
\qquad
u_{n_i} \leftarrow u_{n_i} - \eta\, e_i\, v_c \quad \forall i
```

`embeddings` (W_in) accumulate the gradient across every context word in the window and receive **one update per center position**:

```math
v_c \leftarrow v_c - \eta \sum_{\text{context pairs}} \frac{\partial \mathcal{L}}{\partial v_c}
```

The learning rate decays linearly to a floor of `0.0001`:

```math
\eta_t = \eta_0 \cdot \max\!\left(0.0001,\; 1 - \frac{t}{T}\right)
```

where $`t`$ is the global step count across all epochs and $`T = \text{epochs} \times |\text{corpus}|`$.

---

### Negative sampling distribution

Noise words are drawn from the smoothed unigram distribution:

```math
P(w) \propto \text{count}(w)^{0.75}
```

The exponent 0.75 compresses the frequency gap between common and rare words — frequent words are still sampled more often, but their dominance is reduced. This is the value reported by Mikolov et al. (2013) as empirically optimal.

---

### Subsampling of frequent words

Before training, each token is discarded independently with probability:

```math
P(\text{discard} \mid w) = 1 - \sqrt{\frac{t \cdot N}{f(w)}}
```

where $`f(w)`$ is the absolute frequency of $`w`$, $`N`$ is the total corpus size, and $`t = 10^{-5}`$ is the threshold. Tokens below the threshold frequency are never discarded; tokens far above it are dropped with high probability. The retained sequence is then used as-is, which effectively widens the context window for surviving words.

---

## Testing

The test suite covers the core building blocks: `sigmoid`, `unigram_noise`, `build_vocab`, `subsample`, `SkipGramModel` (initialisation and a single gradient step), and `nearest_words`.

**Install pytest** (if not already installed):

```bash
pip install pytest
```

**Run all tests:**

```bash
pytest tests/ -v
```

**Expected output:**

```
tests/test_word2vec.py::TestSigmoid::test_zero_returns_half          PASSED
tests/test_word2vec.py::TestSigmoid::test_output_range               PASSED
...
35 passed in 0.61s
```

---

## Conclusion


---

## References

- Mikolov et al., [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) (2013)
- Mikolov et al., [*Distributed Representations of Words and Phrases and their Compositionality*](https://arxiv.org/abs/1310.4546) (2013)
- Goldberg & Levy, [*word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*](https://arxiv.org/abs/1402.3722) (2014)

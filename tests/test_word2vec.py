"""
Unit tests for the Word2Vec NumPy implementation.

Run with:
    pytest tests/test_word2vec.py -v
"""

import numpy as np
import pytest

from src.utils import sigmoid, unigram_noise
from src.preprocess import build_vocab, subsample
from src.skipgram_model import SkipGramModel
from src.train import nearest_words


# ---------------------------------------------------------------------------
# Fixtures – small, reusable test data
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_corpus():
    """Minimal corpus: 5 unique words, each appearing multiple times."""
    return ["dog", "cat", "dog", "cat", "fish", "dog", "bird", "cat", "fish", "bird"]


@pytest.fixture
def vocab(tiny_corpus):
    """word2id, id2word and freq_arr built from tiny_corpus."""
    return build_vocab(tiny_corpus, min_count=1)


@pytest.fixture
def model():
    """Small model (vocab=10, dim=8) with a fixed random seed."""
    return SkipGramModel(vocab_size=10, embed_dim=8, seed=42)


# ===========================================================================
# 1. sigmoid
# ===========================================================================

class TestSigmoid:
    def test_zero_returns_half(self):
        """sigmoid(0) should equal exactly 0.5."""
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_output_range(self):
        """All output values must lie in (0, 1)."""
        x = np.linspace(-100, 100, 500)
        result = sigmoid(x)
        assert np.all(result > 0) and np.all(result < 1)

    def test_clipping_prevents_overflow(self):
        """Very large/small inputs should not produce NaN or inf."""
        x = np.array([-1e9, 1e9])
        result = sigmoid(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_monotone_increasing(self):
        """Sigmoid is a strictly increasing function."""
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = sigmoid(x)
        assert np.all(np.diff(result) > 0)

    def test_symmetry(self):
        """sigmoid(-x) == 1 - sigmoid(x) for all x."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        assert sigmoid(-x) == pytest.approx(1 - sigmoid(x))


# ===========================================================================
# 2. unigram_noise
# ===========================================================================

class TestUnigramNoise:
    def test_sums_to_one(self):
        """The probability distribution must sum to 1."""
        freqs = np.array([10.0, 5.0, 1.0, 100.0])
        dist = unigram_noise(freqs)
        assert dist.sum() == pytest.approx(1.0)

    def test_all_nonnegative(self):
        """No probability may be negative."""
        freqs = np.array([1.0, 2.0, 3.0])
        dist = unigram_noise(freqs)
        assert np.all(dist >= 0)

    def test_rare_words_boosted(self):
        """A rare word should have a higher share in the distribution
        than raw frequency alone would give it."""
        freqs = np.array([100.0, 1.0])      # word 0 is 100x more frequent
        dist = unigram_noise(freqs)
        raw_ratio = freqs[0] / freqs[1]     # 100
        smooth_ratio = dist[0] / dist[1]    # < 100 after 0.75 smoothing
        assert smooth_ratio < raw_ratio

    def test_uniform_input_stays_uniform(self):
        """Equal frequencies should produce a uniform distribution."""
        freqs = np.array([5.0, 5.0, 5.0])
        dist = unigram_noise(freqs)
        assert dist == pytest.approx([1/3, 1/3, 1/3])


# ===========================================================================
# 3. build_vocab
# ===========================================================================

class TestBuildVocab:
    def test_returns_three_elements(self, tiny_corpus):
        result = build_vocab(tiny_corpus, min_count=1)
        assert len(result) == 3   # word2id, id2word, freq_arr

    def test_min_count_filtering(self):
        """Words appearing fewer times than min_count are filtered out."""
        tokens = ["rare", "common", "common", "common"]
        word2id, _, _ = build_vocab(tokens, min_count=2)
        assert "rare" not in word2id
        assert "common" in word2id

    def test_single_char_words_filtered(self):
        """Single-character tokens are always dropped."""
        tokens = ["a", "b", "cat", "cat"]
        word2id, _, _ = build_vocab(tokens, min_count=1)
        assert "a" not in word2id
        assert "b" not in word2id
        assert "cat" in word2id

    def test_word2id_and_id2word_are_consistent(self, vocab):
        word2id, id2word, _ = vocab
        for word, idx in word2id.items():
            assert id2word[idx] == word

    def test_ids_are_contiguous(self, vocab):
        """Word IDs form a contiguous range 0..vocab_size-1."""
        word2id, _, _ = vocab
        ids = sorted(word2id.values())
        assert ids == list(range(len(ids)))

    def test_freq_arr_matches_vocab_size(self, vocab):
        word2id, _, freq_arr = vocab
        assert len(freq_arr) == len(word2id)

    def test_freq_arr_values_correct(self, tiny_corpus):
        """Counts in freq_arr must match actual token frequencies."""
        word2id, _, freq_arr = build_vocab(tiny_corpus, min_count=1)
        assert freq_arr[word2id["dog"]] == 3   # "dog" appears 3 times
        assert freq_arr[word2id["cat"]] == 3
        assert freq_arr[word2id["fish"]] == 2


# ===========================================================================
# 4. subsample
# ===========================================================================

class TestSubsample:
    def test_output_is_subset(self):
        """Subsampling can only remove IDs, never introduce new ones."""
        corpus_ids = np.array([0, 1, 0, 1, 2, 0, 2, 1])
        freq_arr   = np.array([4.0, 3.0, 2.0])
        result = subsample(corpus_ids, freq_arr, threshold=1e-5)
        assert set(result).issubset(set(corpus_ids))

    def test_very_high_threshold_keeps_most(self):
        """A very high threshold means almost nothing is discarded."""
        np.random.seed(0)
        corpus_ids = np.arange(100) % 5
        freq_arr   = np.ones(5)
        result = subsample(corpus_ids, freq_arr, threshold=1.0)
        assert len(result) == len(corpus_ids)

    def test_returns_numpy_array(self):
        corpus_ids = np.array([0, 1, 2, 0, 1])
        freq_arr   = np.array([2.0, 2.0, 1.0])
        result = subsample(corpus_ids, freq_arr)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# 5. SkipGramModel.__init__
# ===========================================================================

class TestSkipGramModelInit:
    def test_embedding_shape(self):
        m = SkipGramModel(vocab_size=50, embed_dim=16)
        assert m.embeddings.shape == (50, 16)

    def test_context_weights_shape(self):
        m = SkipGramModel(vocab_size=50, embed_dim=16)
        assert m.context_weights.shape == (50, 16)

    def test_weights_are_small(self):
        """Weights are initialised from N(0, 0.1) — typical values should be < 0.5."""
        m = SkipGramModel(vocab_size=200, embed_dim=64)
        assert np.abs(m.embeddings).max() < 0.5
        assert np.abs(m.context_weights).max() < 0.5

    def test_reproducible_with_same_seed(self):
        m1 = SkipGramModel(vocab_size=20, embed_dim=4, seed=99)
        m2 = SkipGramModel(vocab_size=20, embed_dim=4, seed=99)
        np.testing.assert_array_equal(m1.embeddings, m2.embeddings)

    def test_different_seeds_give_different_weights(self):
        m1 = SkipGramModel(vocab_size=20, embed_dim=4, seed=1)
        m2 = SkipGramModel(vocab_size=20, embed_dim=4, seed=2)
        assert not np.allclose(m1.embeddings, m2.embeddings)


# ===========================================================================
# 6. SkipGramModel.step
# ===========================================================================

class TestSkipGramModelStep:
    def test_loss_is_positive(self, model):
        """NEG loss should always be positive."""
        center_vec = model.embeddings[0].copy()
        negatives  = np.array([2, 3, 4, 5, 6])
        loss, _ = model.step(center_vec, context_id=1, negatives=negatives, lr=0.01)
        assert loss > 0

    def test_gradient_shape(self, model):
        """The gradient must have the same shape as an embedding vector."""
        center_vec = model.embeddings[0].copy()
        negatives  = np.array([2, 3, 4])
        _, grad = model.step(center_vec, context_id=1, negatives=negatives, lr=0.01)
        assert grad.shape == (model.embeddings.shape[1],)

    def test_context_weights_updated(self, model):
        """After calling step, context weights for the positive word must change."""
        context_id = 1
        before = model.context_weights[context_id].copy()
        center_vec = model.embeddings[0].copy()
        negatives  = np.array([3, 4, 5, 6, 7])
        model.step(center_vec, context_id=context_id, negatives=negatives, lr=0.1)
        assert not np.allclose(model.context_weights[context_id], before)

    def test_returns_tuple_of_two(self, model):
        center_vec = model.embeddings[0].copy()
        negatives  = np.array([2, 3])
        result = model.step(center_vec, context_id=1, negatives=negatives, lr=0.01)
        assert isinstance(result, tuple) and len(result) == 2

    def test_loss_decreases_after_many_steps(self):
        """Repeated updates on the same pair should drive the loss down."""
        model = SkipGramModel(vocab_size=10, embed_dim=8, seed=0)
        negatives = np.array([3, 4, 5, 6, 7])
        losses = []
        for _ in range(200):
            center_vec = model.embeddings[0].copy()
            loss, grad = model.step(center_vec, context_id=1, negatives=negatives, lr=0.05)
            model.embeddings[0] -= 0.05 * grad
            losses.append(loss)
        assert losses[-1] < losses[0], "Loss should decrease after many steps"


# ===========================================================================
# 7. nearest_words
# ===========================================================================

class TestNearestWords:
    @pytest.fixture
    def small_embeddings(self):
        """4 words, 3-D embeddings. 'apple' is close to 'fruit'."""
        W = np.array([
            [1.0,  0.0, 0.0],   # apple
            [0.9,  0.1, 0.0],   # fruit   <- very close to apple
            [0.0,  1.0, 0.0],   # dog
            [0.0,  0.0, 1.0],   # house
        ])
        word2id = {"apple": 0, "fruit": 1, "dog": 2, "house": 3}
        id2word = {0: "apple", 1: "fruit", 2: "dog", 3: "house"}
        return W, word2id, id2word

    def test_returns_list(self, small_embeddings):
        W, word2id, id2word = small_embeddings
        result = nearest_words("apple", W, word2id, id2word, top_n=2)
        assert isinstance(result, list)

    def test_correct_top_neighbour(self, small_embeddings):
        """The nearest word to 'apple' should be 'fruit'."""
        W, word2id, id2word = small_embeddings
        result = nearest_words("apple", W, word2id, id2word, top_n=3)
        top_word = result[0][0]
        assert top_word == "fruit"

    def test_query_word_not_in_results(self, small_embeddings):
        """The queried word itself must not appear among the results."""
        W, word2id, id2word = small_embeddings
        result = nearest_words("apple", W, word2id, id2word, top_n=3)
        words = [w for w, _ in result]
        assert "apple" not in words

    def test_unknown_word_returns_empty(self, small_embeddings):
        """A word not in the vocabulary should return an empty list."""
        W, word2id, id2word = small_embeddings
        result = nearest_words("unknown_xyz", W, word2id, id2word, top_n=3)
        assert result == []

    def test_similarity_scores_in_range(self, small_embeddings):
        """Cosine similarity values must lie in [-1, 1]."""
        W, word2id, id2word = small_embeddings
        result = nearest_words("apple", W, word2id, id2word, top_n=3)
        for _, score in result:
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_results_sorted_descending(self, small_embeddings):
        """Results must be sorted from most to least similar."""
        W, word2id, id2word = small_embeddings
        result = nearest_words("apple", W, word2id, id2word, top_n=3)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

import numpy as np
from utils import sigmoid


class SkipGramModel:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 77):
        rng = np.random.default_rng(seed)
        self.embeddings      = rng.normal(0, 0.1, (vocab_size, embed_dim))
        self.context_weights = rng.normal(0, 0.1, (vocab_size, embed_dim))
        self.rng = rng

    def step(
        self,
        center_vec: np.ndarray,
        context_id: int,
        negatives: np.ndarray,
        lr: float,
    ) -> tuple:
        u_pos = self.context_weights[context_id]
        u_neg = self.context_weights[negatives]

        sigma_pos = sigmoid(np.dot(center_vec, u_pos))
        sigma_neg = sigmoid(u_neg @ center_vec)

        loss = -np.log(sigma_pos + 1e-7) - np.sum(np.log(1 - sigma_neg + 1e-7))

        self.context_weights[context_id] -= lr * (sigma_pos - 1) * center_vec
        self.context_weights[negatives]  -= lr * sigma_neg[:, None] * center_vec

        grad = (sigma_pos - 1) * u_pos + (sigma_neg[:, None] * u_neg).sum(axis=0)
        return loss, grad

    def fit_epoch(
        self,
        corpus_ids: np.ndarray,
        noise_dist: np.ndarray,
        window: int,
        n_neg: int,
        base_lr: float,
        epoch: int,
        total_epochs: int,
        total_steps: int,
    ):
        n = len(corpus_ids)
        epoch_offset = (epoch - 1) * n
        log_interval = max(1, n // 10)
        running_loss, n_pairs = 0.0, 0

        print(f"\n{'='*52}")
        print(f"  Epoch {epoch} / {total_epochs}")
        print(f"{'='*52}")

        for step, pos in enumerate(self.rng.permutation(n)):
            lr_t = base_lr * max(0.0001, 1.0 - (epoch_offset + step) / total_steps)

            center_id  = corpus_ids[pos]
            center_vec = self.embeddings[center_id].copy()

            radius    = self.rng.integers(1, window + 1)
            neighbors = corpus_ids[max(0, pos - radius) : pos + radius + 1]

            grad_acc = np.zeros(self.embeddings.shape[1])

            for offset, context_id in enumerate(neighbors):
                if max(0, pos - radius) + offset == pos:
                    continue
                negatives = self.rng.choice(len(noise_dist), size=n_neg, p=noise_dist)
                loss, g = self.step(center_vec, context_id, negatives, lr_t)
                grad_acc     += g
                running_loss += loss
                n_pairs      += 1

            self.embeddings[center_id] -= lr_t * grad_acc

            if (step + 1) % log_interval == 0:
                pct      = (step + 1) / n
                bar_len  = 30
                filled   = int(bar_len * pct)
                bar      = "#" * filled + "-" * (bar_len - filled)
                avg_loss = running_loss / n_pairs
                print(f"  [{bar}] {100*pct:5.1f}%  loss={avg_loss:.4f}  lr={lr_t:.5f}")

        print(f"  done — avg loss: {running_loss/n_pairs:.4f}")

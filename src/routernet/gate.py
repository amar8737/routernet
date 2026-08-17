"""Gating networks for routernet.

* :class:`GatingNetwork` - a small MLP trained with the built-in autograd engine
  that directly optimizes the ensemble cross-entropy loss, producing per-sample
  specialist weights.
* :class:`ConfidenceGate` - a learned per-sample gate deciding when the gated
  ensemble should be trusted over a uniform (global) fallback ensemble.
* :class:`MetaWeightLearner` - a lightweight scikit-learn based weight learner
  used by the regression path.
"""

from __future__ import annotations

import numpy as np

from .autograd import Adam, Parameter, Tensor, temperature_softmax

__all__ = ["GatingNetwork", "ConfidenceGate", "MetaWeightLearner"]


class GatingNetwork:
    """Gating network that directly optimizes ensemble cross-entropy loss.

    Uses the built-in autograd engine for backpropagation.
    """

    def __init__(
        self,
        context_dim: int = 16,
        n_specialists: int = 5,
        hidden_dims: list[int] = (128, 64),
    ):
        self.context_dim = context_dim
        self.n_specialists = n_specialists
        self.hidden_dims = list(hidden_dims)
        self.temperature = 1.0
        self.training = True
        self.dropout = 0.1

        # Layer 1: context_dim -> hidden_dims[0]
        scale1 = np.sqrt(2.0 / context_dim)
        self.W1 = Parameter(
            np.random.randn(context_dim, hidden_dims[0]).astype(np.float32) * scale1,
            "W1",
        )
        self.b1 = Parameter(np.zeros(hidden_dims[0], dtype=np.float32), "b1")
        self.ln1_gamma = Parameter(np.ones(hidden_dims[0], dtype=np.float32), "ln1_gamma")
        self.ln1_beta = Parameter(np.zeros(hidden_dims[0], dtype=np.float32), "ln1_beta")

        # Layer 2: hidden_dims[0] -> hidden_dims[1]
        scale2 = np.sqrt(2.0 / hidden_dims[0])
        self.W2 = Parameter(
            np.random.randn(hidden_dims[0], hidden_dims[1]).astype(np.float32) * scale2,
            "W2",
        )
        self.b2 = Parameter(np.zeros(hidden_dims[1], dtype=np.float32), "b2")
        self.ln2_gamma = Parameter(np.ones(hidden_dims[1], dtype=np.float32), "ln2_gamma")
        self.ln2_beta = Parameter(np.zeros(hidden_dims[1], dtype=np.float32), "ln2_beta")

        # Layer 3: hidden_dims[1] -> n_specialists
        scale3 = np.sqrt(2.0 / hidden_dims[1])
        self.W3 = Parameter(
            np.random.randn(hidden_dims[1], n_specialists).astype(np.float32) * scale3,
            "W3",
        )
        self.b3 = Parameter(np.zeros(n_specialists, dtype=np.float32), "b3")

        # Collect all parameters
        self.parameters = [
            self.W1,
            self.b1,
            self.ln1_gamma,
            self.ln1_beta,
            self.W2,
            self.b2,
            self.ln2_gamma,
            self.ln2_beta,
            self.W3,
            self.b3,
        ]

    def forward(self, context: np.ndarray, training: bool = True):
        self.training = training
        x = Tensor(context) if isinstance(context, np.ndarray) else context

        # Layer 1
        h1 = x @ self.W1 + self.b1
        h1 = h1.layer_norm(self.ln1_gamma, self.ln1_beta)
        h1 = h1.relu()
        if training:
            h1 = h1.dropout(self.dropout)

        # Layer 2
        h2 = h1 @ self.W2 + self.b2
        h2 = h2.layer_norm(self.ln2_gamma, self.ln2_beta)
        h2 = h2.relu()
        if training:
            h2 = h2.dropout(self.dropout)

        # Layer 3
        logits = h2 @ self.W3 + self.b3
        weights = temperature_softmax(logits, self.temperature)
        return weights, logits

    def loss(self, weights: Tensor, oof_probas: np.ndarray, y: np.ndarray):
        """Compute ensemble cross-entropy loss + utilization regularization.

        Built entirely from ``Tensor`` ops so gradients flow back to the gate
        network parameters.
        """
        n, K, C = oof_probas.shape

        # Ensemble probability: sum_k w_k * p_k  (n, C)
        w = weights.unsqueeze(-1)  # (n, K, 1)
        oof_t = Tensor(oof_probas.astype(np.float32), requires_grad=False)
        ensemble = (w * oof_t).sum(axis=1)  # (n, C)

        # Cross-entropy loss (differentiable through log/one-hot masking)
        eps = 1e-9
        one_hot = np.zeros((n, C), dtype=np.float32)
        one_hot[np.arange(n), y] = 1.0
        ce = -((Tensor(one_hot, requires_grad=False) * (ensemble + eps).log()).sum(axis=1)).mean()

        # Utilization regularization: entropy of mean weights
        util = weights.mean(axis=0)  # (K,)
        util_entropy = -((util + 1e-10).log() * util).sum()
        balance_loss = (-0.01) * util_entropy

        total_loss = ce + balance_loss
        ce_np = float(
            -np.mean(np.log(np.clip(ensemble.data[np.arange(n), y], 1e-9, None)))
        )
        return total_loss, ce_np, float(util_entropy.data)

    def fit(
        self,
        context: np.ndarray,
        oof_probas: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        temp_start: float = 1.0,
        temp_end: float = 0.1,
        util_weight: float = 0.01,
        verbose: bool = False,
    ) -> GatingNetwork:
        """Train the gating network using the built-in autograd engine."""
        opt = Adam(self.parameters, lr=lr, weight_decay=1e-5)

        temp_schedule = np.linspace(temp_start, temp_end, epochs)

        for epoch in range(epochs):
            self.temperature = temp_schedule[epoch]

            weights, logits = self.forward(context, training=True)
            loss_tensor, ce, util_ent = self.loss(weights, oof_probas, y)

            opt.zero_grad()
            loss_tensor.backward()
            opt.step()

            if verbose and epoch % 50 == 0:
                print(
                    f"  Gate epoch {epoch}: CE={ce:.4f}, "
                    f"UtilEnt={util_ent:.4f}, Temp={self.temperature:.3f}"
                )

        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict specialist weights for given contexts."""
        weights, _ = self.forward(context, training=False)
        return weights.data


class ConfidenceGate:
    """Learned gate for global fallback."""

    def __init__(self, context_dim: int = 16):
        self.context_dim = context_dim
        self.W = Parameter(
            np.random.randn(context_dim, 1).astype(np.float32) * 0.01, "conf_W"
        )
        self.b = Parameter(np.zeros(1, dtype=np.float32), "conf_b")
        self.parameters = [self.W, self.b]

    def forward(self, context: np.ndarray) -> Tensor:
        x = Tensor(context) if isinstance(context, np.ndarray) else context
        logits = x @ self.W + self.b
        logits = logits * 0.5  # dampen scale before sigmoid
        return 1.0 / (1.0 + (-logits).exp())  # sigmoid

    def fit(
        self,
        context: np.ndarray,
        oof_probas: np.ndarray,
        y: np.ndarray,
        gate_weights: np.ndarray | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> ConfidenceGate:
        """Train to predict when the gated ensemble beats the global (uniform) ensemble.

        The target is +1 when the gated routing is correct while the uniform
        ensemble is wrong, 0 for the opposite, and 0.5 when they agree.
        """
        n = len(y)
        n_specialists = oof_probas.shape[1]

        if gate_weights is None:
            gate_weights = np.full((n, n_specialists), 1.0 / n_specialists)

        gated_proba = np.sum(gate_weights[:, :, None] * oof_probas, axis=1)
        global_proba = np.mean(oof_probas, axis=1)

        gated_correct = np.argmax(gated_proba, axis=1) == y
        global_correct = np.argmax(global_proba, axis=1) == y

        target = np.where(
            gated_correct & ~global_correct,
            1.0,
            np.where(global_correct & ~gated_correct, 0.0, 0.5),
        ).astype(np.float32).reshape(-1, 1)

        opt = Adam(self.parameters, lr=lr)

        for epoch in range(epochs):
            context_tensor = Tensor(context)
            gate_logits = context_tensor @ self.W + self.b
            gate = Tensor(1.0) / (Tensor(1.0) + (-gate_logits * 0.5).exp())

            # BCE loss
            eps = 1e-15
            target_tensor = Tensor(target)
            loss = -(
                target_tensor * (gate + eps).log()
                + (Tensor(1.0) - target_tensor) * (Tensor(1.0) - gate + eps).log()
            ).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if verbose and epoch % 50 == 0:
                print(f"  ConfGate epoch {epoch}: BCE={loss.data.item():.4f}")

        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        x = Tensor(context) if isinstance(context, np.ndarray) else context
        logits = x @ self.W + self.b
        gate = Tensor(1.0) / (Tensor(1.0) + (-logits * 0.5).exp())
        return gate.data


class MetaWeightLearner:
    """Learns specialist weights from context embeddings using MLPRegressor."""

    def __init__(
        self,
        context_dim: int = 16,
        n_specialists: int = 3,
        hidden_dim: int = 16,
        random_state: int = 42,
    ):
        self.context_dim = context_dim
        self.n_specialists = n_specialists
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.model = None

    def fit(self, context: np.ndarray, specialist_scores: np.ndarray) -> MetaWeightLearner:
        """Fit the weight learner to per-sample specialist scores.

        Args:
            context: Context embeddings (n_samples, context_dim)
            specialist_scores: Performance of each specialist per sample
                               (n_samples, n_specialists)

        Returns:
            self
        """
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim,),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=self.random_state,
        )
        specialist_scores = np.clip(specialist_scores, 0, None)
        norm_scores = specialist_scores / (
            np.sum(specialist_scores, axis=1, keepdims=True) + 1e-8
        )
        self.model.fit(context, norm_scores)
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict specialist weights, normalized to sum to 1 per sample."""
        scores = self.model.predict(context)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

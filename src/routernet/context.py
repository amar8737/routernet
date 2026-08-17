"""Context encoder for routernet.

Learns a 16-dimensional context embedding for each sample that captures local
neighborhood geometry, feature statistics, label structure and specialist
agreement. These embeddings are the inputs to the gating network that routes
each sample to the specialists best suited for it.

Feature Schema (16 features, indices 0-15):
    0:  mean_knn_dist        - Mean distance to k-nearest neighbors
    1:  std_knn_dist         - Std of distances to k-nearest neighbors
    2:  sample_feat_var      - Per-sample feature variance
    3:  mahalanobis          - Mahalanobis-like distance from center
    4:  local_feat_var       - Mean feature variance within k-NN neighborhood
    5:  min_knn_dist         - Distance to nearest neighbor (excl. self)
    6:  max_knn_dist         - Distance to farthest neighbor
    7:  local_density        - 1 / (mean_knn_dist + eps)
    8:  knn_label_entropy    - Entropy of neighbor labels
    9:  knn_class_margin     - Top-1 minus Top-2 class frequency in neighborhood
    10: disagreement_entropy - Entropy of mean specialist predictions (OOF for training)
    11: avg_pred_entropy     - Mean specialist prediction entropy (OOF for training)
    12: max_pred_conf        - Maximum specialist confidence (OOF for training)
    13: decision_margin      - Distance to decision boundary (OOF for training)
    14: global_feat_var      - Global feature variance (precomputed at fit)
    15: global_class_entropy - Global class entropy (precomputed at fit)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_array

__all__ = ["ContextEncoder"]


class ContextEncoder(BaseEstimator, TransformerMixin):
    """Learns a 16-dimensional context embedding for each sample.

    The context is always 16-dimensional; ``context_dim`` is kept only for
    backward compatibility. Neighbor features are read from the fitted KNN
    model's stored data rather than keeping a second copy of ``X``, so the
    trained encoder does not retain the raw training dataset.
    """

    FEATURE_NAMES: list[str] = [
        "mean_knn_dist",
        "std_knn_dist",
        "sample_feat_var",
        "mahalanobis",
        "local_feat_var",
        "min_knn_dist",
        "max_knn_dist",
        "local_density",
        "knn_label_entropy",
        "knn_class_margin",
        "disagreement_entropy",
        "avg_pred_entropy",
        "max_pred_conf",
        "decision_margin",
        "global_feat_var",
        "global_class_entropy",
    ]

    def __init__(self, n_neighbors: int = 10, context_dim: int = 16):
        self.n_neighbors = n_neighbors
        self.context_dim = context_dim
        self.n_features_out_ = len(self.FEATURE_NAMES)
        self.scaler = StandardScaler()
        self.context_scaler = RobustScaler()
        self.knn_model: NearestNeighbors | None = None
        self.feature_stats: dict | None = None
        self.y_train_: np.ndarray | None = None
        self.oof_probas_: np.ndarray | None = None
        self.fitted_specialists_: list | None = None
        self._neighbor_data: np.ndarray | None = None
        self.global_feat_var_: float = 0.0
        self.global_class_entropy_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        oof_probas: np.ndarray | None = None,
        fitted_specialists: list | None = None,
    ) -> ContextEncoder:
        """Fit the context encoder.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,) - needed for label entropy features
            oof_probas: OOF specialist probabilities
                (n_samples, n_specialists, n_classes) for training
            fitted_specialists: List of fitted specialist models for inference features

        Returns:
            self
        """
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.y_train_ = y
        self.oof_probas_ = oof_probas
        self.fitted_specialists_ = fitted_specialists

        # Scale features for KNN
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # KNN with k+1 neighbors, exclude self
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(X)), metric="euclidean"
        )
        self.knn_model.fit(X_scaled)
        # Reference (no copy) to the data KNN already stores internally
        self._neighbor_data = self.knn_model._fit_X

        # Feature-level statistics
        self.feature_stats = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0) + 1e-8,
            "entropy": self._compute_entropy(X),
        }

        # Precompute global scalars so we never need to retain X
        self.global_feat_var_ = float(np.mean(np.var(X, axis=0)))
        if y is not None:
            vals, counts = np.unique(y, return_counts=True)
            probs = counts / counts.sum()
            self.global_class_entropy_ = float(
                -np.sum(probs * np.log2(probs + 1e-10))
            )
        else:
            self.global_class_entropy_ = 0.0

        # Compute context on training data to fit context scaler
        sample_context = self._compute_context(X, X_scaled, oof_probas, fitted_specialists)
        self.context_scaler.fit(sample_context)
        return self

    def transform(
        self,
        X: np.ndarray,
        oof_probas: np.ndarray | None = None,
        fitted_specialists: list | None = None,
    ) -> np.ndarray:
        """Transform samples to context embeddings.

        Args:
            X: Feature matrix (n_samples, n_features)
            oof_probas: OOF probabilities for training (features 10-13)
            fitted_specialists: Fitted specialists for inference (features 10-13)

        Returns:
            Context embeddings (n_samples, n_features_out_)
        """
        X = check_array(X, accept_sparse=False)
        X_scaled = self.scaler.transform(X)
        context = self._compute_context(X, X_scaled, oof_probas, fitted_specialists)
        context = self.context_scaler.transform(context)
        return context

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        oof_probas: np.ndarray | None = None,
        fitted_specialists: list | None = None,
    ) -> np.ndarray:
        """Fit and transform in one call (keeps routernet-specific kwargs)."""
        return self.fit(
            X, y=y, oof_probas=oof_probas, fitted_specialists=fitted_specialists
        ).transform(X, oof_probas=oof_probas, fitted_specialists=fitted_specialists)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Feature names for Pipeline output."""
        return np.asarray(self.FEATURE_NAMES, dtype=object)

    def _compute_context(
        self,
        X: np.ndarray,
        X_scaled: np.ndarray,
        oof_probas: np.ndarray | None,
        fitted_specialists: list | None,
    ) -> np.ndarray:
        """Compute all 16 context features."""
        n_samples = X.shape[0]

        # KNN on scaled features
        distances, indices = self.knn_model.kneighbors(X_scaled)
        # Exclude self (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        context = np.zeros((n_samples, self.n_features_out_))

        # Feature 0: Mean KNN distance
        context[:, 0] = np.mean(distances, axis=1)

        # Feature 1: Std of KNN distances
        context[:, 1] = np.std(distances, axis=1)

        # Feature 2: Sample feature variance
        context[:, 2] = np.var(X, axis=1)

        # Feature 3: Mahalanobis-like distance
        context[:, 3] = np.linalg.norm(
            (X - self.feature_stats["mean"]) / self.feature_stats["std"], axis=1
        )

        # Feature 4: Local feature variance (within k-NN neighborhood)
        if self._neighbor_data is not None:
            neighbor_feats = self._neighbor_data[indices]  # (n, k, d)
            context[:, 4] = np.mean(np.var(neighbor_feats, axis=1), axis=1)
        else:
            context[:, 4] = np.mean(np.var(X, axis=0))

        # Feature 5: Min KNN distance
        context[:, 5] = distances[:, 0]

        # Feature 6: Max KNN distance
        context[:, 6] = distances[:, -1]

        # Feature 7: Local density
        context[:, 7] = 1.0 / (context[:, 0] + 1e-8)

        # Features 8-9: Label-based (need y_train)
        if self.y_train_ is not None:
            neighbor_labels = self.y_train_[indices]  # (n, k)

            for i in range(n_samples):
                vals, counts = np.unique(neighbor_labels[i], return_counts=True)
                probs = counts / counts.sum()
                context[i, 8] = -np.sum(probs * np.log2(probs + 1e-10))
                sorted_probs = np.sort(probs)[::-1]
                context[i, 9] = sorted_probs[0] - (
                    sorted_probs[1] if len(sorted_probs) > 1 else 0
                )
        else:
            context[:, 8] = 0
            context[:, 9] = 0

        # Features 10-13: Specialist-based (OOF for training, fitted for inference)
        if oof_probas is not None:
            self._compute_specialist_context(context, oof_probas, n_samples)
        elif fitted_specialists is not None:
            self._compute_specialist_context_from_fitted(
                context, X, fitted_specialists, n_samples
            )
        else:
            context[:, 10:14] = 0

        # Features 14-15: Global statistics (precomputed at fit)
        context[:, 14] = self.global_feat_var_
        context[:, 15] = self.global_class_entropy_

        return context

    def _compute_specialist_context(
        self, context: np.ndarray, oof_probas: np.ndarray, n_samples: int
    ) -> None:
        """Compute specialist-based context features from OOF probabilities.

        Features:
            10: disagreement_entropy - Entropy of mean specialist predictions
            11: avg_pred_entropy - Mean specialist prediction entropy
            12: max_pred_conf - Maximum specialist confidence
            13: decision_margin - Margin from decision boundary (prob margin)
        """
        mean_probas = np.mean(oof_probas, axis=1)

        for i in range(n_samples):
            probs = mean_probas[i]
            probs = probs / (probs.sum() + 1e-10)
            context[i, 10] = -np.sum(probs * np.log2(probs + 1e-10))

        for i in range(n_samples):
            entropies = []
            for k in range(oof_probas.shape[1]):
                probs = oof_probas[i, k]
                probs = probs / (probs.sum() + 1e-10)
                entropies.append(-np.sum(probs * np.log2(probs + 1e-10)))
            context[i, 11] = np.mean(entropies)

        for i in range(n_samples):
            max_confs = [np.max(oof_probas[i, k]) for k in range(oof_probas.shape[1])]
            context[i, 12] = np.max(max_confs)

        for i in range(n_samples):
            probs = mean_probas[i]
            probs = probs / (probs.sum() + 1e-10)
            sorted_probs = np.sort(probs)[::-1]
            context[i, 13] = sorted_probs[0] - (
                sorted_probs[1] if len(sorted_probs) > 1 else 0
            )

    def _compute_specialist_context_from_fitted(
        self,
        context: np.ndarray,
        X: np.ndarray,
        fitted_specialists: list,
        n_samples: int,
    ) -> None:
        """Compute specialist-based context from fitted specialists (inference)."""
        specialist_probas = []
        for spec in fitted_specialists:
            if not hasattr(spec, "predict_proba"):
                continue
            specialist_probas.append(spec.predict_proba(X))

        if not specialist_probas:
            context[:, 10:14] = 0
            return

        # (K, n, C) -> (n, K, C)
        specialist_probas = np.array(specialist_probas).transpose(1, 0, 2)
        self._compute_specialist_context(context, specialist_probas, n_samples)

    @staticmethod
    def _compute_entropy(X: np.ndarray) -> np.ndarray:
        """Compute feature-level entropy (how dispersed each feature is)."""
        entropy = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            if np.var(X[:, j]) == 0:
                entropy[j] = 0.0
                continue
            hist, _ = np.histogram(X[:, j], bins=10)
            hist = hist / np.sum(hist)
            entropy[j] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        return entropy

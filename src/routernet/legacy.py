"""Legacy v1 estimators (backwards compatibility).

The original routernet implementation routes samples using a simpler
8-dimensional context and a hand-rolled meta weight learner trained on
per-sample specialist accuracy. Kept for reproducibility of earlier
experiments; new projects should prefer :class:`routernet.RouternetClassifier`
and :class:`routernet.RouternetRegressor`.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_X_y

__all__ = [
    "LegacyContextEncoder",
    "LegacyMetaWeightLearner",
    "LegacyRouternetClassifier",
    "LegacyRouternetRegressor",
]


class LegacyContextEncoder:
    """Learns a context embedding for each sample based on K-NN geometry and
    feature statistics (the original 8-dimensional v1 schema)."""

    def __init__(self, n_neighbors: int = 5, context_dim: int = 8):
        self.n_neighbors = n_neighbors
        self.context_dim = context_dim
        self.scaler = StandardScaler()
        self.knn_model = None
        self.feature_stats = None

    def fit(self, X, base_models=None):
        X = check_array(X, accept_sparse=False)
        self.X_train_ = X.copy()
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(X))
        )
        self.knn_model.fit(X)
        self.feature_stats = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "entropy": self._compute_entropy(X),
        }
        self.n_features = X.shape[1]
        return self

    def transform(self, X, base_models=None):
        X = check_array(X, accept_sparse=False)
        n_samples = X.shape[0]
        context = np.zeros((n_samples, self.context_dim))

        distances, indices = self.knn_model.kneighbors(X)

        context[:, 0] = np.mean(distances, axis=1)
        context[:, 1] = np.std(distances, axis=1)
        context[:, 2] = self._sample_entropy(X)
        context[:, 3] = np.linalg.norm(
            (X - self.feature_stats["mean"]) / (self.feature_stats["std"] + 1e-8),
            axis=1,
        )

        for i in range(n_samples):
            context[i, 4] = np.mean(np.var(X, axis=0))
            context[i, 5] = np.min(distances[i]) if len(distances[i]) > 0 else 0
            context[i, 6] = np.max(distances[i]) if len(distances[i]) > 0 else 1
            context[i, 7] = self.context_dim / 2

        return np.clip(context, 0, 1)

    @staticmethod
    def _compute_entropy(X):
        entropy = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            hist, _ = np.histogram(X[:, j], bins=10)
            hist = hist / np.sum(hist)
            entropy[j] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        return entropy

    @staticmethod
    def _sample_entropy(X):
        return np.var(X, axis=1)


class LegacyMetaWeightLearner:
    """Learns to predict specialist weights given a context embedding using a
    simple two-layer network trained with gradient descent."""

    def __init__(self, context_dim: int = 8, n_specialists: int = 3, hidden_dim: int = 16):
        self.context_dim = context_dim
        self.n_specialists = n_specialists
        self.hidden_dim = hidden_dim
        self.W1 = self.b1 = self.W2 = self.b2 = None

    def fit(self, context, specialist_scores):
        specialist_scores = np.clip(specialist_scores, 0, 1)
        norm_scores = specialist_scores / (
            np.sum(specialist_scores, axis=1, keepdims=True) + 1e-8
        )

        np.random.seed(42)
        self.W1 = np.random.randn(self.context_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.n_specialists) * 0.01
        self.b2 = np.zeros((1, self.n_specialists))

        learning_rate = 0.1
        for _ in range(100):
            h = np.maximum(0, np.dot(context, self.W1) + self.b1)
            logits = np.dot(h, self.W2) + self.b2
            weights = self._softmax(logits)

            dlogits = weights - norm_scores
            dW2 = np.dot(h.T, dlogits) / len(context)
            db2 = np.sum(dlogits, axis=0, keepdims=True) / len(context)
            dh = np.dot(dlogits, self.W2.T)
            dh = dh * (h > 0)
            dW1 = np.dot(context.T, dh) / len(context)
            db1 = np.sum(dh, axis=0, keepdims=True) / len(context)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
        return self

    def predict(self, context):
        h = np.maximum(0, np.dot(context, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        return self._softmax(logits)

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LegacyRouternetClassifier(BaseEstimator, ClassifierMixin):
    """Original v1 routernet classifier (legacy). See module docstring."""

    def __init__(
        self,
        n_specialists: int = 3,
        context_dim: int = 8,
        n_neighbors: int = 5,
        specialist_types: list | None = None,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.n_specialists = n_specialists
        self.context_dim = context_dim
        self.n_neighbors = n_neighbors
        self.specialist_types = specialist_types or ["gb", "rf", "svm"][:n_specialists]
        self.verbose = verbose
        self.random_state = random_state

    def _log(self, msg):
        if self.verbose:
            print(f"[routernet-legacy] {msg}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_encoded = np.searchsorted(self.classes_, y)

        self._log(f"Training {self.n_specialists} specialist models...")
        self.specialists_ = []
        for i in range(self.n_specialists):
            specialist = self._create_specialist(i)
            specialist.fit(X, y_encoded)
            self.specialists_.append(specialist)

        self._log("Training context encoder...")
        self.context_encoder_ = LegacyContextEncoder(
            n_neighbors=self.n_neighbors, context_dim=self.context_dim
        )
        self.context_encoder_.fit(X)
        context = self.context_encoder_.transform(X)

        self._log("Computing specialist performances...")
        specialist_scores = np.zeros((len(X), self.n_specialists))
        for i, specialist in enumerate(self.specialists_):
            preds = specialist.predict(X)
            specialist_scores[:, i] = (preds == y_encoded).astype(float)

        self._log("Training meta-weight learner...")
        self.weight_learner_ = LegacyMetaWeightLearner(
            context_dim=self.context_dim, n_specialists=self.n_specialists
        )
        self.weight_learner_.fit(context, specialist_scores)

        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(X)
        weights = self.weight_learner_.predict(context)

        specialist_probas = []
        for specialist in self.specialists_:
            if hasattr(specialist, "predict_proba"):
                specialist_probas.append(specialist.predict_proba(X))
            else:
                preds = specialist.predict(X)
                proba = np.zeros((len(X), self.n_classes_))
                for i, pred in enumerate(preds):
                    proba[i, pred] = 1.0
                specialist_probas.append(proba)

        final_proba = np.zeros((len(X), self.n_classes_))
        for i in range(len(X)):
            for j, specialist_proba in enumerate(specialist_probas):
                final_proba[i] += weights[i, j] * specialist_proba[i]
        return final_proba

    def get_specialist_weights(self, X):
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(X)
        return self.weight_learner_.predict(context)

    def get_context(self, X):
        X = check_array(X, accept_sparse=False)
        return self.context_encoder_.transform(X)

    def _create_specialist(self, idx):
        if isinstance(self.specialist_types[idx], str):
            spec_type = self.specialist_types[idx].lower()
            rs = self.random_state + idx
            if spec_type == "gb":
                return GradientBoostingClassifier(
                    n_estimators=50, max_depth=5, learning_rate=0.1, random_state=rs
                )
            elif spec_type == "rf":
                return RandomForestClassifier(
                    n_estimators=50, max_depth=7, random_state=rs
                )
            elif spec_type == "svm":
                return SVC(kernel="rbf", probability=True, random_state=rs)
            elif spec_type == "mlp":
                return MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=200, random_state=rs
                )
            elif spec_type == "dt":
                return DecisionTreeClassifier(max_depth=10, random_state=rs)
            raise ValueError(f"Unknown specialist type: {spec_type}")
        return self.specialist_types[idx]


class LegacyRouternetRegressor(BaseEstimator, RegressorMixin):
    """Original v1 routernet regressor (legacy). See module docstring."""

    def __init__(
        self,
        n_specialists: int = 3,
        context_dim: int = 8,
        n_neighbors: int = 5,
        specialist_types: list | None = None,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.n_specialists = n_specialists
        self.context_dim = context_dim
        self.n_neighbors = n_neighbors
        self.specialist_types = specialist_types or ["gb", "rf", "mlp"][:n_specialists]
        self.verbose = verbose
        self.random_state = random_state

    def _log(self, msg):
        if self.verbose:
            print(f"[routernet-legacy] {msg}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        np.random.seed(self.random_state)

        self._log(f"Training {self.n_specialists} specialist models...")
        self.specialists_ = []
        for i in range(self.n_specialists):
            specialist = self._create_specialist(i)
            specialist.fit(X, y)
            self.specialists_.append(specialist)

        self._log("Training context encoder...")
        self.context_encoder_ = LegacyContextEncoder(
            n_neighbors=self.n_neighbors, context_dim=self.context_dim
        )
        self.context_encoder_.fit(X)
        context = self.context_encoder_.transform(X)

        self._log("Computing specialist performances...")
        specialist_scores = np.zeros((len(X), self.n_specialists))
        for i, specialist in enumerate(self.specialists_):
            preds = specialist.predict(X)
            mse = np.mean((preds - y) ** 2, axis=1) if len(y.shape) > 1 else (preds - y) ** 2
            specialist_scores[:, i] = 1.0 / (1.0 + mse)

        self._log("Training meta-weight learner...")
        self.weight_learner_ = LegacyMetaWeightLearner(
            context_dim=self.context_dim, n_specialists=self.n_specialists
        )
        self.weight_learner_.fit(context, specialist_scores)

        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(X)
        weights = self.weight_learner_.predict(context)

        specialist_preds = [s.predict(X) for s in self.specialists_]
        final_preds = np.zeros(len(X))
        for i in range(len(X)):
            for j, specialist_pred in enumerate(specialist_preds):
                final_preds[i] += weights[i, j] * specialist_pred[i]
        return final_preds

    def get_specialist_weights(self, X):
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(X)
        return self.weight_learner_.predict(context)

    def _create_specialist(self, idx):
        if isinstance(self.specialist_types[idx], str):
            spec_type = self.specialist_types[idx].lower()
            rs = self.random_state + idx
            if spec_type == "gb":
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(
                    n_estimators=50, max_depth=5, learning_rate=0.1, random_state=rs
                )
            elif spec_type == "rf":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    n_estimators=50, max_depth=7, random_state=rs
                )
            elif spec_type == "mlp":
                return MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=200, random_state=rs
                )
            raise ValueError(f"Unknown specialist type: {spec_type}")
        return self.specialist_types[idx]

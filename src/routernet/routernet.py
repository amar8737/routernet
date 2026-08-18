"""Routernet estimators.

A per-sample specialist routing ensemble with learned context awareness. The
classifiers and regressors in this module combine a pool of diverse specialist
models with a learned gating network that routes each sample to the specialists
best suited for it, based on a learned context embedding.

Key design points:
- Out-of-fold (OOF) specialist predictions train the gate, avoiding leakage.
- The gating network directly optimizes the ensemble cross-entropy loss.
- A confidence gate provides a per-sample global fallback to the uniform
  ensemble when routing would be unreliable.
- Scikit-learn compatible (``BaseEstimator``), so it works in ``Pipeline``s,
  with ``GridSearchCV`` and with OpenML's ``run_model_on_task``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .context import ContextEncoder
from .gate import ConfidenceGate, GatingNetwork, MetaWeightLearner

__all__ = ["RouternetClassifier", "RouternetRegressor"]


class RouternetClassifier(BaseEstimator, ClassifierMixin):
    """Per-sample specialist routing ensemble classifier.

    Combines multiple specialist classifiers with context-aware dynamic
    weighting. Each sample is routed to specialists based on learned context
    embeddings. Uses OOF specialist probabilities to train a gating network
    that directly optimizes the ensemble cross-entropy loss.

    Parameters
    ----------
    n_specialists : int, default=5
        Number of specialist models to train.
    context_dim : int, default=16
        Dimensionality of context embeddings (the encoder always produces 16
        features; kept for API compatibility).
    n_neighbors : int, default=10
        Number of neighbors for KNN in the context encoder.
    specialist_types : list of str or estimators, default=['cat', 'xgb', 'rf', 'svm', 'mlp']
        Types of specialist models. String options: 'gb', 'rf', 'svm', 'mlp',
        'dt', 'cat', 'xgb', 'et'. Alternatively, provide pre-instantiated
        sklearn estimators (they are cloned in ``fit``).
    oof_folds : int, default=5
        Number of folds for out-of-fold specialist predictions.
    use_global_fallback : bool, default=True
        Whether to use confidence-gated global fallback.
    gate_hidden : tuple of int, default=(128, 64)
        Hidden dimensions of the gating network.
    gate_epochs : int, default=200
        Number of epochs to train the gating network.
    gate_lr : float, default=1e-3
        Learning rate for the gating network.
    verbose : bool, default=False
        Whether to print progress during fitting.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_specialists: int = 5,
        context_dim: int = 16,
        n_neighbors: int = 10,
        specialist_types: list | None = None,
        oof_folds: int = 5,
        use_global_fallback: bool = True,
        gate_hidden: tuple = (128, 64),
        gate_epochs: int = 200,
        gate_lr: float = 1e-3,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.n_specialists = n_specialists
        self.context_dim = context_dim
        self.n_neighbors = n_neighbors
        self.specialist_types = specialist_types or [
            "cat",
            "xgb",
            "rf",
            "svm",
            "mlp",
        ][:n_specialists]
        self.oof_folds = oof_folds
        self.use_global_fallback = use_global_fallback
        self.gate_hidden = gate_hidden
        self.gate_epochs = gate_epochs
        self.gate_lr = gate_lr
        self.verbose = verbose
        self.random_state = random_state

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[routernet] {msg}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> RouternetClassifier:
        """Fit the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]

        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Encode labels to 0..C-1
        y_encoded = np.searchsorted(self.classes_, y)

        # 1. Train specialists on full data (for final inference)
        self._log(f"Training {self.n_specialists} specialists...")
        self.specialists_ = []
        for i in range(self.n_specialists):
            specialist = self._create_specialist(i)
            specialist.fit(X, y_encoded)
            self.specialists_.append(specialist)
            self._log(f"  Specialist {i + 1}/{self.n_specialists} done")

        # 2. Compute OOF probabilities for gate training
        oof_folds = self._effective_oof_folds(y_encoded)
        self._log(f"Computing OOF probabilities ({oof_folds}-fold)...")
        oof_probas = self._compute_oof_probas(X, y_encoded, n_folds=oof_folds)

        # 3. Context encoder (with OOF for features 10-13 during training)
        self._log("Fitting context encoder...")
        self.context_encoder_ = ContextEncoder(
            n_neighbors=self.n_neighbors,
            context_dim=self.context_dim,
        )
        self.context_encoder_.fit(
            X,
            y=y_encoded,
            oof_probas=oof_probas,
            fitted_specialists=self.specialists_,
        )
        context = self.context_encoder_.transform(
            X,
            oof_probas=oof_probas,
            fitted_specialists=self.specialists_,
        )

        # 4. Gating network (direct CE on OOF)
        self._log("Training gating network (direct CE)...")
        self.gate_ = GatingNetwork(
            context_dim=self.context_dim,
            n_specialists=self.n_specialists,
            hidden_dims=list(self.gate_hidden),
        )
        self.gate_.fit(
            context,
            oof_probas,
            y_encoded,
            epochs=self.gate_epochs,
            lr=self.gate_lr,
            verbose=self.verbose,
        )

        # 5. Confidence gate for global fallback (trained after the gate)
        if self.use_global_fallback:
            self._log("Training confidence gate...")
            gate_weights = self.gate_.predict(context)
            self.conf_gate_ = ConfidenceGate(context_dim=self.context_dim)
            self.conf_gate_.fit(
                context,
                oof_probas,
                y_encoded,
                gate_weights=gate_weights,
                verbose=self.verbose,
            )

        return self

    def _effective_oof_folds(self, y_encoded: np.ndarray) -> int:
        """Cap ``oof_folds`` so StratifiedKFold can never fail on small classes."""
        _, counts = np.unique(y_encoded, return_counts=True)
        max_viable = max(int(counts.min()), 2)
        return max(2, min(self.oof_folds, max_viable))

    def _compute_oof_probas(
        self, X: np.ndarray, y: np.ndarray, n_folds: int
    ) -> np.ndarray:
        """Return (n_samples, n_specialists, n_classes) OOF probabilities."""
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )
        n_samples = len(X)
        oof_probas = np.zeros((n_samples, self.n_specialists, self.n_classes_))

        for _, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr = y[tr_idx]

            for k in range(self.n_specialists):
                specialist = self._create_specialist(k)
                specialist.fit(X_tr, y_tr)
                proba = specialist.predict_proba(X_val)
                oof_probas[val_idx, k] = proba

        return oof_probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Context with fitted specialists for features 10-13
        context = self.context_encoder_.transform(
            X, fitted_specialists=self.specialists_
        )

        # Gate weights
        weights = self.gate_.predict(context)  # (n, K)

        # Specialist probabilities
        specialist_probas = np.stack(
            [self._get_proba(spec, X) for spec in self.specialists_], axis=1
        )  # (n, K, C)

        # Ensemble: sum_k w_k * p_k
        ensemble_proba = np.sum(
            weights[:, :, np.newaxis] * specialist_probas, axis=1
        )

        if self.use_global_fallback:
            global_proba = np.mean(specialist_probas, axis=1)
            gate = self.conf_gate_.predict(context)  # (n, 1)
            final_proba = gate * ensemble_proba + (1 - gate) * global_proba
        else:
            final_proba = ensemble_proba

        return final_proba

    def get_specialist_weights(self, X: np.ndarray) -> np.ndarray:
        """Get the specialist weights for each sample in X.

        Useful for interpretability. Returns (n_samples, n_specialists).
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(
            X, fitted_specialists=self.specialists_
        )
        return self.gate_.predict(context)

    def get_context(self, X: np.ndarray) -> np.ndarray:
        """Get the context embeddings for samples in X.

        Returns (n_samples, n_features_out) with the 16-feature context schema.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.context_encoder_.transform(
            X, fitted_specialists=self.specialists_
        )

    def _create_specialist(self, idx: int):
        """Create a specialist model based on type.

        Accepts either a string type name or a pre-instantiated estimator
        (which is cloned so user-provided objects are never mutated).
        """
        spec = self.specialist_types[idx]
        if not isinstance(spec, str):
            return clone(spec)

        spec_type = spec.lower()
        rs = self.random_state + idx

        if spec_type == "cat":
            try:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.05,
                    random_seed=self.random_state + idx,
                    verbose=False,
                    thread_count=-1,
                )
            except ImportError:
                return ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=rs,
                    n_jobs=-1,
                )
        elif spec_type == "xgb":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    n_jobs=-1,
                    tree_method="hist",
                    eval_metric="logloss",
                )
            except ImportError:
                return GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=rs,
                )
        elif spec_type == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=rs,
                n_jobs=-1,
            )
        elif spec_type == "et":
            return ExtraTreesClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=rs,
                n_jobs=-1,
            )
        elif spec_type == "svm":
            return CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=rs),
                ensemble=False,
            )
        elif spec_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                alpha=0.001,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=rs,
            )
        elif spec_type == "gb":
            return GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=rs,
            )
        elif spec_type == "dt":
            return DecisionTreeClassifier(max_depth=10, random_state=rs)
        else:
            raise ValueError(f"Unknown specialist type: {spec_type}")

    @staticmethod
    def _get_proba(specialist, X: np.ndarray) -> np.ndarray:
        """Safely get probabilities from a specialist."""
        if hasattr(specialist, "predict_proba"):
            return specialist.predict_proba(X)
        # Convert predictions to probabilities
        preds = specialist.predict(X)
        n_classes = np.max(preds) + 1
        proba = np.zeros((len(X), n_classes))
        for i, pred in enumerate(preds):
            proba[i, pred] = 1.0
        return proba


class RouternetRegressor(BaseEstimator, RegressorMixin):
    """Per-sample specialist routing ensemble regressor.

    Regression version of routernet. Adapts the classifier logic for continuous
    targets using a meta weight learner over specialist predictions.

    Parameters
    ----------
    n_specialists : int, default=3
        Number of specialist models.
    context_dim : int, default=16
        Context embedding dimensionality (context is always 16-dimensional).
    n_neighbors : int, default=5
        KNN neighbors for context encoding.
    specialist_types : list, default=['gb', 'rf', 'mlp']
        Specialist model types for regression (strings or estimator instances).
    verbose : bool, default=False
        Whether to print progress during fitting.
    random_state : int, default=42
        Random seed.
    """

    def __init__(
        self,
        n_specialists: int = 3,
        context_dim: int = 16,
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

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[routernet] {msg}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> RouternetRegressor:
        """Fit the regressor."""
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]

        np.random.seed(self.random_state)

        self._log(f"Training {self.n_specialists} specialist models...")
        self.specialists_ = []
        for i in range(self.n_specialists):
            specialist = self._create_specialist(i)
            specialist.fit(X, y)
            self.specialists_.append(specialist)
            self._log(f"  Specialist {i + 1}/{self.n_specialists} trained")

        # Train context encoder
        self._log("Training context encoder...")
        self.context_encoder_ = ContextEncoder(
            n_neighbors=self.n_neighbors, context_dim=self.context_dim
        )
        self.context_encoder_.fit(X)
        context = self.context_encoder_.transform(X)

        # Compute specialist performance (inverse MSE) per sample
        self._log("Computing specialist performances...")
        specialist_scores = np.zeros((len(X), self.n_specialists))
        for i, specialist in enumerate(self.specialists_):
            preds = specialist.predict(X)
            mse = (preds - y) ** 2
            specialist_scores[:, i] = 1.0 / (1.0 + mse)

        # Train weight learner
        self._log("Training meta-weight learner...")
        self.weight_learner_ = MetaWeightLearner(
            context_dim=self.context_dim,
            n_specialists=self.n_specialists,
            random_state=self.random_state,
        )
        self.weight_learner_.fit(context, specialist_scores)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        context = self.context_encoder_.transform(
            X, fitted_specialists=self.specialists_
        )
        weights = self.weight_learner_.predict(context)  # (n, K)

        specialist_preds = np.stack(
            [specialist.predict(X) for specialist in self.specialists_], axis=1
        )  # (n, K)

        return np.sum(weights * specialist_preds, axis=1)

    def get_specialist_weights(self, X: np.ndarray) -> np.ndarray:
        """Get specialist weights for interpretability."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        context = self.context_encoder_.transform(
            X, fitted_specialists=self.specialists_
        )
        return self.weight_learner_.predict(context)

    def get_context(self, X: np.ndarray) -> np.ndarray:
        """Get the context embeddings for samples in X."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.context_encoder_.transform(X)

    def _create_specialist(self, idx: int):
        """Create a specialist for regression (string type or estimator instance)."""
        spec = self.specialist_types[idx]
        if not isinstance(spec, str):
            return clone(spec)

        spec_type = spec.lower()
        rs = self.random_state + idx

        if spec_type == "gb":
            return GradientBoostingRegressor(
                n_estimators=50, max_depth=5, learning_rate=0.1, random_state=rs
            )
        elif spec_type == "rf":
            return RandomForestRegressor(
                n_estimators=50, max_depth=7, random_state=rs
            )
        elif spec_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=200, random_state=rs
            )
        else:
            raise ValueError(f"Unknown specialist type: {spec_type}")

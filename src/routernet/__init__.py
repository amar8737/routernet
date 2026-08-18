"""routernet: per-sample specialist routing ensemble with learned context awareness.

A routing ensemble that combines a pool of diverse specialist models with a
learned gating network. Each sample is routed to the specialists best suited
for it based on a learned 16-dimensional context embedding that captures
neighborhood geometry, feature statistics, label structure and specialist
agreement.

Example
-------
>>> from routernet import RouternetClassifier
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>> X, y = make_classification(n_samples=300, n_features=20, random_state=42)
>>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
>>> clf = RouternetClassifier(n_specialists=3, random_state=42)
>>> clf.fit(X_tr, y_tr)
>>> accuracy_score(y_te, clf.predict(X_te)) > 0.5
True
"""

from .context import ContextEncoder
from .gate import ConfidenceGate, GatingNetwork, MetaWeightLearner
from .routernet import RouternetClassifier, RouternetRegressor
from .utils import analyze_routing, print_routing_analysis

__version__ = "0.1.2"

__all__ = [
    "RouternetClassifier",
    "RouternetRegressor",
    "ContextEncoder",
    "GatingNetwork",
    "ConfidenceGate",
    "MetaWeightLearner",
    "analyze_routing",
    "print_routing_analysis",
    "__version__",
]

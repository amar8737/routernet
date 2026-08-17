"""Utilities for interpreting routernet routing decisions."""

from __future__ import annotations

import numpy as np

__all__ = ["analyze_routing", "print_routing_analysis"]


def analyze_routing(
    model, X: np.ndarray, y: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
) -> dict:
    """Analyze how samples are routed to specialists.

    Args:
        model: Fitted ``RouternetClassifier`` or ``RouternetRegressor``.
        X: Feature matrix.
        y: Target labels/values (optional, currently informational).
        sample_indices: Specific indices to analyze (default: first 10).

    Returns:
        Dictionary with routing analysis.
    """
    if sample_indices is None:
        sample_indices = np.arange(min(10, len(X)))

    weights = model.get_specialist_weights(X[sample_indices])
    context = model.get_context(X[sample_indices])

    return {
        "sample_indices": sample_indices,
        "weights": weights,
        "context": context,
        "primary_specialist": np.argmax(weights, axis=1),
        "weight_entropy": -np.sum(weights * np.log(weights + 1e-10), axis=1),
    }


def print_routing_analysis(analysis: dict) -> None:
    """Pretty print routing analysis."""
    print("\n" + "=" * 70)
    print("ROUTERNET ROUTING ANALYSIS")
    print("=" * 70)

    for idx in range(len(analysis["sample_indices"])):
        sample_idx = analysis["sample_indices"][idx]
        weights = analysis["weights"][idx]
        primary = analysis["primary_specialist"][idx]
        entropy = analysis["weight_entropy"][idx]

        print(f"\nSample {sample_idx}:")
        print(f"  Primary specialist: {primary}")
        print(f"  Specialist weights: {weights.round(3)}")
        print(f"  Weight entropy (uncertainty): {entropy:.3f}")
        print(f"  Context features: {analysis['context'][idx].round(2)}")

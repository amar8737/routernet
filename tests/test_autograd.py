"""Tests for the built-in autograd engine."""

import numpy as np
import pytest

from routernet.autograd import (
    Adam,
    Parameter,
    Tensor,
    cross_entropy_loss,
    temperature_softmax,
)


def test_linear_gradient():
    """Gradient of (x @ W + b) w.r.t. x matches numpy autodiff."""
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(4, 3)).astype(np.float32))
    W = Parameter(rng.normal(size=(3, 2)).astype(np.float32), "W")
    b = Parameter(rng.normal(size=(2,)).astype(np.float32), "b")

    y = (x @ W + b).mean()
    y.backward()

    assert x.grad.shape == x.data.shape
    assert W.grad.shape == W.data.shape
    assert b.grad.shape == b.data.shape

    # Numeric check on W
    for idx in [(0, 0), (1, 1), (2, 0)]:
        eps = 1e-4
        Wp = W.data.copy()
        Wm = W.data.copy()
        Wp[idx] += eps
        Wm[idx] -= eps
        yp = np.mean(x.data @ Wp + b.data)
        ym = np.mean(x.data @ Wm + b.data)
        numeric = (yp - ym) / (2 * eps)
        assert np.isclose(W.grad[idx], numeric, atol=1e-3)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(1)
    logits = Tensor(rng.normal(size=(5, 4)).astype(np.float32))
    probs = logits.softmax()
    assert np.allclose(probs.data.sum(axis=1), 1.0, atol=1e-6)


def test_temperature_softmax():
    rng = np.random.default_rng(2)
    logits = Tensor(rng.normal(size=(3, 3)).astype(np.float32))
    cold = temperature_softmax(logits, 0.1).data
    hot = temperature_softmax(logits, 10.0).data
    assert np.allclose(cold.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(hot.sum(axis=1), 1.0, atol=1e-6)
    # Cold temperature should be more peaked
    assert np.max(cold) > np.max(hot)


def test_cross_entropy_decreases_with_adam():
    rng = np.random.default_rng(3)
    W = Parameter(rng.normal(size=(4, 3)).astype(np.float32) * 0.1, "W")
    x_data = rng.normal(size=(16, 4)).astype(np.float32)
    targets = rng.integers(0, 3, size=16)

    opt = Adam([W], lr=1e-1)
    losses = []
    for _ in range(30):
        logits = Tensor(x_data) @ W
        loss = cross_entropy_loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.data.item())

    assert losses[-1] < losses[0]
    # Should get close to the best possible log-loss for the fit
    assert losses[-1] < 1.05


def test_layer_norm_shape_and_finite():
    rng = np.random.default_rng(4)
    x = Tensor(rng.normal(size=(8, 5)).astype(np.float32))
    w = Parameter(np.ones(5, dtype=np.float32), "w")
    b = Parameter(np.zeros(5, dtype=np.float32), "b")
    out = x.layer_norm(w, b).relu().mean()
    out.backward()
    assert np.isfinite(x.grad).all()
    assert x.grad.shape == x.data.shape


def test_dropout_inference_is_identity():
    rng = np.random.default_rng(5)
    x = Tensor(rng.normal(size=(4, 4)).astype(np.float32))
    x2 = x.dropout(0.5, training=False)
    assert np.allclose(x2.data, x.data)


@pytest.mark.parametrize("op", ["add", "mul", "sub", "div"])
def test_elementwise_gradients(op):
    rng = np.random.default_rng(6)
    a = Parameter(rng.normal(size=(3,)).astype(np.float32), "a")
    b = Parameter(rng.normal(size=(3,)).astype(np.float32), "b")
    if op == "add":
        out = (a + b).sum()
    elif op == "mul":
        out = (a * b).sum()
    elif op == "sub":
        out = (a - b).sum()
    else:
        out = (a / b).sum()
    out.backward()
    assert np.isfinite(a.grad).all()
    assert np.isfinite(b.grad).all()
    assert not np.allclose(a.grad, 0)
    assert not np.allclose(b.grad, 0)

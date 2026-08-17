"""Micrograd-style autograd engine for routernet.

A minimal automatic differentiation engine supporting:
- Basic ops: ``+``, ``-``, ``*``, ``/``, matmul
- Activations: relu, softmax, exp, log
- Normalization: layer_norm
- Regularization: dropout
- Losses: cross_entropy
- Optimizer: Adam

All operations are vectorized over numpy arrays (``Tensor``) so the gating
networks train quickly on CPU without external deep-learning frameworks.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Value",
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "LayerNorm",
    "Dropout",
    "Sequential",
    "Adam",
    "cross_entropy_loss",
    "temperature_softmax",
    "utilization_regularization",
]


class Value:
    """Single scalar value with gradient."""

    def __init__(
        self,
        data: float,
        _children: tuple = (),
        _op: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only scalar powers supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Run reverse-mode autodiff from this scalar."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


class Tensor:
    """Multi-dimensional array with autograd support."""

    def __init__(
        self,
        data: np.ndarray | float | int,
        _children: tuple = (),
        _op: str = "",
        label: str = "",
        requires_grad: bool = True,
    ):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)
        else:
            data = np.asarray(data, dtype=np.float32)
        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.requires_grad = requires_grad
        self._shape = self.data.shape

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    @staticmethod
    def _broadcast_grad(param_shape: tuple, grad: np.ndarray) -> np.ndarray:
        """Sum a gradient over dimensions that were broadcast."""
        orig_ndim = len(param_shape)
        grad_shape = grad.shape
        if len(param_shape) < len(grad_shape):
            param_shape = (1,) * (len(grad_shape) - len(param_shape)) + param_shape
        for i, (s, g) in enumerate(zip(param_shape, grad_shape, strict=True)):
            if s == 1 and g > 1:
                axis = len(grad_shape) - len(param_shape) + i
                grad = np.sum(grad, axis=axis, keepdims=True)
        while grad.ndim > orig_ndim:
            grad = np.sum(grad, axis=0)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if self.data.shape != grad.shape:
                    grad = self._broadcast_grad(self.data.shape, grad)
                self.grad += grad
            if other.requires_grad:
                grad = out.grad
                if other.data.shape != grad.shape:
                    grad = self._broadcast_grad(other.data.shape, grad)
                other.grad += grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data - other.data, (self, other), "-")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if self.data.shape != grad.shape:
                    grad = self._broadcast_grad(self.data.shape, grad)
                self.grad += grad
            if other.requires_grad:
                grad = out.grad
                if other.data.shape != grad.shape:
                    grad = self._broadcast_grad(other.data.shape, grad)
                other.grad -= grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                if self.data.shape != grad.shape:
                    grad = self._broadcast_grad(self.data.shape, grad)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                if other.data.shape != grad.shape:
                    grad = self._broadcast_grad(other.data.shape, grad)
                other.grad += grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            if self.requires_grad:
                grad = (1 / other.data) * out.grad
                if self.data.shape != grad.shape:
                    grad = self._broadcast_grad(self.data.shape, grad)
                self.grad += grad
            if other.requires_grad:
                grad = (-self.data / (other.data**2)) * out.grad
                if other.data.shape != grad.shape:
                    grad = self._broadcast_grad(other.data.shape, grad)
                other.grad += grad

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.swapaxes(-2, -1)
            if other.requires_grad:
                other.grad += self.data.swapaxes(-2, -1) @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), f"sum_{axis}")

        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    grad = out.grad
                    if not keepdims:
                        grad = np.expand_dims(grad, axis)
                    self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), f"mean_{axis}")

        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))

        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad / n
                else:
                    grad = out.grad
                    if not keepdims:
                        grad = np.expand_dims(grad, axis)
                    self.grad += grad / n

        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out_data = np.log(np.clip(self.data, 1e-15, None))
        out = Tensor(out_data, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1 / np.clip(self.data, 1e-15, None)) * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """Numerically stable softmax."""
        max_val = np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(self.data - max_val)
        sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
        out_data = exp_data / sum_exp
        out = Tensor(out_data, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                s = out_data
                ds = out.grad
                sum_s_ds = np.sum(s * ds, axis=axis, keepdims=True)
                self.grad += s * (ds - sum_s_ds)

        out._backward = _backward
        return out

    def layer_norm(self, weight: Tensor, bias: Tensor, eps: float = 1e-5):
        """Layer normalization over the last dimension."""
        mean = np.mean(self.data, axis=-1, keepdims=True)
        var = np.var(self.data, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (self.data - mean) * inv_std
        out_data = x_hat * weight.data + bias.data
        out = Tensor(out_data, (self, weight, bias), "layer_norm")

        def _backward():
            if self.requires_grad:
                N = self.data.shape[-1]
                dx_hat = out.grad * weight.data
                dvar = (
                    np.sum(dx_hat * (self.data - mean), axis=-1, keepdims=True)
                    * (-0.5)
                    * inv_std**3
                )
                dmean = np.sum(dx_hat * (-inv_std), axis=-1, keepdims=True) + dvar * np.mean(
                    -2.0 * (self.data - mean), axis=-1, keepdims=True
                )
                dx = dx_hat * inv_std + dvar * 2.0 * (self.data - mean) / N + dmean / N
                self.grad += dx
            if weight.requires_grad:
                weight.grad += np.sum(out.grad * x_hat, axis=tuple(range(out.grad.ndim - 1)))
            if bias.requires_grad:
                bias.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - 1)))

        out._backward = _backward
        return out

    def dropout(self, p: float = 0.1, training: bool = True):
        if not training or p == 0:
            return self
        mask = np.random.binomial(1, 1 - p, size=self.data.shape) / (1 - p)
        out_data = self.data * mask
        out = Tensor(out_data, (self,), "dropout")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def cross_entropy(self, targets: np.ndarray):
        """Cross entropy loss. ``targets``: (n,) class indices."""
        logits = self
        max_val = np.max(logits.data, axis=-1, keepdims=True)
        exp_data = np.exp(logits.data - max_val)
        sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
        log_probs = logits.data - max_val - np.log(sum_exp)
        n = targets.shape[0]
        loss_data = -np.mean(log_probs[np.arange(n), targets])
        out = Tensor(np.array([loss_data]), (logits,), "cross_entropy")

        def _backward():
            if logits.requires_grad:
                probs = exp_data / sum_exp
                dlogits = probs.copy()
                dlogits[np.arange(n), targets] -= 1
                dlogits /= n
                logits.grad += dlogits

        out._backward = _backward
        return out

    def backward(self):
        """Run reverse-mode autodiff from this tensor."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


class Parameter(Tensor):
    """Trainable parameter."""

    def __init__(self, data: np.ndarray, label: str = ""):
        super().__init__(data, label=label, requires_grad=True)


class Module:
    """Base class for neural network modules."""

    def __init__(self):
        self._parameters: list[Parameter] = []

    def parameters(self) -> list[Parameter]:
        return self._parameters

    def register_parameter(self, param: Parameter) -> Parameter:
        self._parameters.append(param)
        return param

    def zero_grad(self):
        for p in self._parameters:
            p.grad = np.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, label: str = ""
    ):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.weight = self.register_parameter(
            Parameter(
                np.random.randn(in_features, out_features).astype(np.float32) * scale,
                f"{label}_weight",
            )
        )
        if bias:
            self.bias = self.register_parameter(
                Parameter(np.zeros(out_features, dtype=np.float32), f"{label}_bias")
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, label: str = ""):
        super().__init__()
        self.eps = eps
        self.weight = self.register_parameter(
            Parameter(np.ones(normalized_shape, dtype=np.float32), f"{label}_ln_weight")
        )
        self.bias = self.register_parameter(
            Parameter(np.zeros(normalized_shape, dtype=np.float32), f"{label}_ln_bias")
        )

    def forward(self, x: Tensor) -> Tensor:
        return x.layer_norm(self.weight, self.bias, self.eps)


class Dropout(Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(self.p, self.training)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        for m in modules:
            for p in m.parameters():
                self.register_parameter(p)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class Adam:
    """Adam optimizer."""

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in parameters}

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.data

            pid = id(p)
            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * p.grad
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (p.grad**2)

            m_hat = self.m[pid] / (1 - self.beta1**self.t)
            v_hat = self.v[pid] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Cross entropy loss. ``logits``: (n, C), ``targets``: (n,)."""
    return logits.cross_entropy(targets)


def temperature_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Softmax with temperature."""
    return (logits / temperature).softmax()


def utilization_regularization(weights: Tensor, weight: float = 0.01) -> Tensor:
    """Entropy of mean weights to encourage specialist utilization."""
    mean_weights = weights.mean(axis=0)  # (K,)
    eps = 1e-10
    entropy = -(mean_weights * (mean_weights + eps).log()).sum()
    return entropy * (-weight)  # Negative entropy to encourage utilization

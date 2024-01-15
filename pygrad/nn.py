"""
This module contains the classes for neural networks and loss functions.
"""

from typing import List, Callable, Any, Optional
from pygrad.core import Tensor
import pygrad as pg
import numpy as np


class Parameter(Tensor):
    """Subclass of Tensor for parameters, has gradients on by default"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('requires_grad', True)
        super().__init__(*args, **kwargs)


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []
    

class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        weight = pg.xavier_uniform(in_features, out_features, nonlinearity="relu", dtype=dtype)
        self.weight = Parameter(weight)

        if self.use_bias:
            b = pg.xavier_uniform(out_features, 1, nonlinearity="relu", dtype=dtype).reshape((1, out_features))
            self.bias = Parameter(b)

    def forward(self, X: Tensor) -> Tensor:
        out = pg.matmul(X, self.weight)
        if self.use_bias:
            bias = pg.broadcast_to(self.bias, out.shape)
            out = pg.add(out, bias)
        return out


class Flatten(Module):
    def forward(self, X):
        n = X.shape[0]
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        return pg.reshape(X, (n, dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return pg.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return pg.sigmoid(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = pg.randb(*x.shape, p= 1 - self.p) / (1 - self.p)
            return x * mask
        return x


class _Loss(Module):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        reduction = reduction.lower()
        valid_reductions = ["none", "mean", "sum"]
        
        if reduction not in valid_reductions:
            raise ValueError(f"Reduction {reduction} not supported, please use one of {valid_reductions}")
        
        self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.weight = weight


class MSELoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)
    

class BCELoss(_WeightedLoss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
    

def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean"
) -> Tensor:
    if input.dtype not in ['float32', 'float64']:
        raise TypeError(f"dtype of input must be float32 or float64, was {input.dtype}")
    
    epsilon = 1e-15  # To prevent log(0)
    input.data = pg.Tensor(np.clip(input.numpy(), epsilon, 1 - epsilon), dtype=input.dtype)
    bce = target * pg.log(input) + (1 - target) * pg.log(1 - input)

    if weight is not None:
        bce = bce * weight

    if reduction == "none":
        return -bce
    elif reduction == "sum":
        return -bce.sum()
    else:
        return -bce.sum() / target.shape[0]


def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean"
) -> Tensor:
    mse = (input - target) * (input - target)

    if reduction == "none":
        return mse
    elif reduction == "sum":
        return mse.sum()
    else:
        return mse.sum() / target.shape[0]
    
"""
Mathematical functions on ND-Tensors.
"""
from typing import Optional

from ..core import Array
from ..core import Function, Tensor

import numpy as np


class EWiseAdd(Function):
    def forward(self, a: Array, b: Array):        
        return a + b

    def backward(self, grad_output: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")
        
        a, b = node.inputs
        return sum_reduce(grad_output, a.shape), sum_reduce(grad_output, b.shape)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: Array):
        return a + self.scalar

    def backward(self, grad_output: Tensor, node: Tensor):
        return grad_output


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(Function):
    def forward(self, a: Array, b: Array):
        return a * b

    def backward(self, grad_output: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs
        return sum_reduce(grad_output * b, a.shape), sum_reduce(grad_output * a, b.shape)


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: Array):
        return a * self.scalar

    def backward(self, grad_output: Tensor, node: Tensor):
        return grad_output * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(Function):
    def forward(self, a: Array, b: Array) -> Array:
        return a ** b

    def backward(self, grad_output, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs
        grad_a = grad_output * b * (a ** (b - 1))
        grad_b = grad_output * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(Function):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def forward(self, a: Array) -> Array:
        return a ** self.scalar

    def backward(self, grad_output, node):
        return grad_output * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(Function):
    def forward(self, a, b):
        return a / b

    def backward(self, grad_output, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")
        
        a, b = node.inputs
        grad_a = grad_output / b
        grad_b = -grad_output * a / (b * b)
        return sum_reduce(grad_a, a.shape), sum_reduce(grad_b, b.shape)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        return a / self.scalar

    def backward(self, grad_output, node):
        return grad_output / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(Function):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a):
        return np.transpose(a, axes=self.axes)

    def backward(self, grad_output, node):
        if self.axes is None:
            return transpose(grad_output)
        inv_axes = np.argsort(self.axes)
        return transpose(grad_output, axes=inv_axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return np.reshape(a, self.shape)

    def backward(self, grad_output, node):
        return reshape(grad_output, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return np.broadcast_to(a, self.shape)

    def backward(self, grad_output, node):
        input_shape = node.inputs[0].shape

        return sum_reduce(grad_output, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(Function):
    def __init__(self, axis: Optional[tuple] = None):
        if axis is not None and not isinstance(axis, tuple):
            axis = (axis,)
        self.axis = axis

    def forward(self, a):
        return np.sum(a, axis=self.axis)

    def backward(self, grad_output, node):
        input_shape = node.inputs[0].shape

        if self.axis is None:
            return grad_output.broadcast_to(input_shape)

        shape = [1 if i in self.axis else s for i, s in enumerate(input_shape)]

        return grad_output.reshape(shape).broadcast_to(input_shape)


def summation(a, axis=None):
    return Summation(axis)(a)


class MatMul(Function):
    def forward(self, a, b):
        return np.matmul(a, b)

    def backward(self, grad_output, node):        
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")
        
        a, b = node.inputs

        # vector dot product
        if a.ndim == 1 and b.ndim == 1:
            return grad_output * b, grad_output * a
        
        if b.ndim == 1:
            grad_a = matmul(reshape(grad_output, (-1,1)), reshape(b, (1,-1)))
        else:
            grad_a = matmul(grad_output, transpose(b))
        
        grad_b = matmul(transpose(a), grad_output)

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(Function):
    def forward(self, a):
        return -a

    def backward(self, grad_output, node):
        return -grad_output


def negate(a):
    return Negate()(a)


class Log(Function):
    def forward(self, a):
        return np.log(a)

    def backward(self, grad_output, node):
        return grad_output / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(Function):
    def forward(self, a):
        return np.exp(a)

    def backward(self, grad_output, node):
        return grad_output * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(Function):
    def forward(self, a):
        return np.maximum(0, a)

    def backward(self, grad_output, node):
        return grad_output * Tensor(node.inputs[0]._data > 0, dtype=grad_output.dtype)


def relu(a):
    return ReLU()(a)


class Sigmoid(Function):
    def forward(self, a):
        self.sigmoid = 1 / (1 + np.exp(-a))
        return self.sigmoid

    def backward(self, grad_output, node):
        return grad_output * self.sigmoid * (1.0 - self.sigmoid)


def sigmoid(a):
    return Sigmoid()(a)


class SumReduce(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        if a.shape == self.shape:
            return a

        if a.ndim != len(self.shape):
            if a.ndim < len(self.shape):
                raise ValueError(
                    f"Gradient of broadcasted operation has lower dimensionality ({a.ndim}) than its input ({len(self.shape)})."
                )
            a = a.sum(axis=tuple(range(a.ndim - len(self.shape))))

        keepdims = tuple(n for n, i in enumerate(a.shape) if i != self.shape[n])
        if keepdims:
            a = a.sum(axis=keepdims, keepdims=True)

        return a
    
    def backward(self, grad_output, node):
        return broadcast_to(grad_output, node.inputs[0].shape)


def sum_reduce(a, shape):
    return SumReduce(shape)(a)

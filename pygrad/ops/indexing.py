"""Indexing-related functions for ND-Tensors"""

from ..core import Array
from ..core import Function, Tensor
from .math import sum_reduce

import numpy as np

class Where(Function):
    def __init__(self, condition):
        if isinstance(condition, Tensor):
            self.condition = condition.numpy().astype(bool)
        else:
            self.condition = np.asarray(condition, dtype=bool)

    def forward(self, a: Array, b: Array):
        return np.where(self.condition, a, b)

    def backward(self, grad_output: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = where(self.condition, grad_output, 0)
        grad_b = where(~self.condition, grad_output, 0)
        return sum_reduce(grad_a, a.shape), sum_reduce(grad_b, b.shape)
    

def where(a, b, condition):
    return Where(condition)(a, b)
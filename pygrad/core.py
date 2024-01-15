"""
This module defines the Tensor class and the Function class.
It implements backprogation over a dynamically built computational graph..
"""

from typing import Dict, List, Union, Tuple, Optional, DefaultDict
from collections import defaultdict
import pygrad
import numpy as np

Array = np.ndarray

class Function:
    def __call__(self, *args, const=False):
        return Tensor.make_from_func(self, args)
    
    def forward(self, *args: Tuple[Array]):
        raise NotImplementedError()
    
    def backward(self, grad_output: "Tensor", node: "Tensor") -> Union["Tensor", Tuple["Tensor"]]:
        raise NotImplementedError()


class Tensor:
    _data: Array
    grad: "Tensor"
    func: Optional[Function]
    requires_grad: bool
    inputs: List["Tensor"]

    def __init__(
        self,
        array,
        *,
        dtype=None,
        func: Optional[Function] = None,
        inputs: List["Tensor"] = [],
        requires_grad: Optional[bool] = None
    ):
        if isinstance(array, Tensor):
            if dtype is None or dtype == array.dtype:
                self._data = array._data
            else:
                self._data = array._data.astype(dtype)
        else:
            self._data = np.array(array, dtype=dtype)
        
        if self._data.ndim == 0:
            self._data = np.expand_dims(self._data,-1)

        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        
        self.func = func
        self.inputs = inputs
        self.requires_grad = requires_grad

    @staticmethod
    def make_from_func(func: Function, inputs: List["Tensor"], const=False):
        input_data = [input_tensor._data for input_tensor in inputs]
        output_data = func.forward(*input_data)
        requires_grad = any(input_tensor.requires_grad for input_tensor in inputs)

        return Tensor(
            array=output_data, 
            dtype=output_data.dtype,
            func=func, 
            inputs=inputs, 
            requires_grad=requires_grad
        )
    
    @staticmethod
    def make_const(data, requires_grad=False):
        return Tensor(
            array=data, 
            dtype=data.dtype,
            requires_grad=requires_grad
        )
    
    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self._data = value._data
    
    def detach(self):
        return Tensor.make_const(self._data)
    
    def is_leaf(self):
        return self.func is None
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def ndim(self):
        return self._data.ndim
    
    def backward(self, grad_output=None):
        """
        if 'self' is a non scalar Tensor,
        i.e. has shape != (1,), 
        backward behaves as if the tensor was summed first 
        (tensor.sum.backward())
        """
        grad_output = (
            grad_output
            if grad_output
            else pygrad.ones(*self.shape, dtype=self.dtype)
        )
        compute_grad(self, grad_output)

    def __repr__(self):
        return "pygrad.Tensor(" + str(self._data) + ")"

    def __str__(self):
        return self._data.__str__()
    
    def numpy(self):
        return self._data
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return pygrad.ops.EWiseAdd()(self, other)
        else:
            return pygrad.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return pygrad.ops.EWiseMul()(self, other)
        else:
            return pygrad.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return pygrad.ops.EWisePow()(self, other)
        else:
            return pygrad.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return pygrad.ops.EWiseAdd()(self, pygrad.ops.Negate()(other))
        else:
            return pygrad.ops.AddScalar(-other)(self)
    
    def __rsub__(self, other):
        return pygrad.ops.AddScalar(other)(pygrad.ops.Negate()(self))

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return pygrad.ops.EWiseDiv()(self, other)
        else:
            return pygrad.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return pygrad.ops.MatMul()(self, other)

    def matmul(self, other):
        return pygrad.ops.MatMul()(self, other)

    def sum(self, axis=None):
        return pygrad.ops.Summation(axis)(self)

    def broadcast_to(self, shape):
        return pygrad.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return pygrad.ops.Reshape(shape)(self)

    def __neg__(self):
        return pygrad.ops.Negate()(self)

    def transpose(self, axis=None):
        return pygrad.ops.Transpose(axis)(self)
    

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__


def compute_grad(tensor_output, grad_output):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    node_to_grad_outputs: DefaultDict[Tensor, List[Tensor]] = defaultdict(list)
    node_to_grad_outputs[tensor_output] = [grad_output]

    reverse_topo_order = list(reversed(toposort([tensor_output])))

    for node in reverse_topo_order:
        grad = sum_node_list(node_to_grad_outputs[node])

        if node.requires_grad:
            node.grad = grad

        if not node.is_leaf():
            parent_grads = node.func.backward(grad, node)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)

            for parent, pgrad in zip(node.inputs, parent_grads):
                node_to_grad_outputs[parent].append(pgrad)


def toposort(node_list: List[Tensor]) -> List[Tensor]:
    """ Topological sort of the nodes for backward pass. """
    visited = set()
    topo_order = []

    def visit(n):
        if n not in visited:
            visited.add(n)
            for parent in n.inputs:
                visit(parent)
            topo_order.append(n)

    for node in node_list:
        visit(node)

    return topo_order


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)

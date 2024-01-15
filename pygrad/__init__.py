"""
PyGrad
======

PyGrad is a Python library for automatic differentiation and neural networks.

It provides a multidimensional Tensor object that supports automatic differentiation and a simple and intuitive API for building and training neural networks.

PyGrad follows NumPy and PyTorch's conventions, making it easy to learn and use.
"""

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "ops",
    "nn",
    "optim",
    "init",
]


from . import ops
from .ops import *
from .core import Tensor
from .init import *
from . import nn
from . import optim
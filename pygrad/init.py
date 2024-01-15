import numpy as np
from .core import Tensor
import math


def empty(*shape, dtype="float32", requires_grad=False):
    array = np.empty(*shape, dtype=dtype)
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def rand(*shape, low=0.0, high=1.0, dtype="float32", requires_grad=False):
    array = np.random.random_sample(shape) * (high - low) + low
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
    array = np.random.standard_normal(shape) * std + mean
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, dtype="float32", requires_grad=False):
    array = np.ones(shape, dtype=dtype) * c
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, dtype="float32", requires_grad=False):
    array = np.ones(shape, dtype=dtype)
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, dtype="float32", requires_grad=False):
    array = np.zeros(shape, dtype=dtype)
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def randb(*shape, p=0.5, dtype="bool", requires_grad=False):
    array = np.random.random_sample(shape) <= p
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, dtype="float32", requires_grad=False):
    return Tensor(
        np.eye(n, dtype=dtype)[i.numpy()],
        requires_grad=requires_grad,
    )


def zeros_like(array, *, requires_grad=False):
    return zeros(
        *array.shape, dtype=array.dtype, requires_grad=requires_grad
    )


def ones_like(array, *, requires_grad=False):
    return ones(
        *array.shape, dtype=array.dtype, requires_grad=requires_grad
    )


def calculate_gain(nonlinearity):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    ================= ====================================================
    """
    if nonlinearity in ['linear', 'sigmoid']:
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_fan_in_and_fan_out(shape):
    from operator import mul
    from functools import reduce

    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[0]
    num_output_fmaps = shape[1]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = reduce(mul, shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def xavier_uniform(*shape, gain=None, nonlinearity="relu", dtype="float32", requires_grad=False):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    gain = calculate_gain(nonlinearity) if gain is None else gain
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return rand(fan_in, fan_out, low=-a, high=a, dtype=dtype, requires_grad=requires_grad)


def xavier_normal(*shape, gain=None, nonlinearity="relu", dtype="float32", requires_grad=False):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    gain = calculate_gain(nonlinearity) if gain is None else gain
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, dtype=dtype, requires_grad=requires_grad)


def kaiming_uniform(*shape, mode='fan_in', nonlinearity="relu", dtype="float32", requires_grad=False):
    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return rand(*shape, low=-bound, high=bound, dtype=dtype, requires_grad=requires_grad)


def kaiming_normal(*shape, mode='fan_in', nonlinearity="relu", dtype="float32", requires_grad=False):
    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    return randn(*shape, mean=0.0, std=std, dtype=dtype, requires_grad=requires_grad)
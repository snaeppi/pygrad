"""
Optimizers for pygrad.
"""

import pygrad as pg

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad_wd = param.grad.detach() + self.weight_decay * param.detach()
            new_u = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * grad_wd
            new_u = pg.Tensor(new_u, dtype=param.dtype)
            self.u[param] = new_u 
            param.data -= self.lr * new_u 


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params: 
            grad_wd = param.grad.detach() + self.weight_decay * param.detach()

            new_m = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad_wd
            new_v = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * grad_wd * grad_wd

            self.m[param] = new_m
            self.v[param] = new_v

            m_with_bias_corr = (new_m / (1 - self.beta1 ** self.t)).detach()
            v_with_bias_corr = (new_v / (1 - self.beta2 ** self.t)).detach()

            update = self.lr * (m_with_bias_corr) / (v_with_bias_corr ** 0.5 + self.eps)
            update = pg.Tensor(update, dtype=param.dtype)

            param.data -= update.detach()

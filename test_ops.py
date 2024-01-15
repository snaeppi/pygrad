import unittest

import pygrad as pg
import numpy as np

class TestMathOps(unittest.TestCase):

    def test_ewise_add(self):
        a = pg.Tensor(3.0, requires_grad=True)
        b = pg.Tensor(4.0, requires_grad=True)
        c = pg.add(a, b)
        c.backward()

        self.assertEqual(c.numpy().item(), 7.0)
        self.assertEqual(a.grad.numpy().item(), 1.0)
        self.assertEqual(b.grad.numpy().item(), 1.0)

    def test_add_scalar(self):
        a = pg.Tensor(5.0, requires_grad=True)
        c = pg.add_scalar(a, 2.0)
        c.backward()

        self.assertEqual(c.numpy().item(), 7.0)
        self.assertEqual(a.grad.numpy().item(), 1.0)

    def test_ewise_mul(self):
        a = pg.Tensor(3.0, requires_grad=True)
        b = pg.Tensor(4.0, requires_grad=True)
        c = pg.multiply(a, b)
        c.backward()

        self.assertEqual(c.numpy().item(), 12.0)
        self.assertEqual(a.grad.numpy().item(), 4.0)
        self.assertEqual(b.grad.numpy().item(), 3.0)

    def test_mul_scalar(self):
        a = pg.Tensor(6.0, requires_grad=True)
        c = pg.mul_scalar(a, 3.0)
        c.backward()

        self.assertEqual(c.numpy().item(), 18.0)
        self.assertEqual(a.grad.numpy().item(), 3.0)

    def test_ewise_pow(self):
        a = pg.Tensor(2.0, requires_grad=True)
        b = pg.Tensor(3.0, requires_grad=True)
        c = pg.power(a, b)
        c.backward()

        self.assertEqual(c.numpy().item(), 8.0)
        self.assertEqual(a.grad.numpy().item(), 3.0*2.0**(3.0-1))
        self.assertEqual(b.grad.numpy().item(), 2.0**3.0 * np.log(2.0)) 

    def test_pow_scalar(self):
        a = pg.Tensor(4.0, requires_grad=True)
        c = pg.power_scalar(a, 2)
        c.backward()

        self.assertEqual(c.numpy().item(), 16.0)
        self.assertEqual(a.grad.numpy().item(), 8.0)

    def test_ewise_div(self):
        a = pg.Tensor(2.0, requires_grad=True)
        b = pg.Tensor(4.0, requires_grad=True)
        c = a / b
        c.backward()

        self.assertEqual(c.numpy().item(), 0.5)
        self.assertEqual(a.grad.numpy().item(), 1 / 4.0) 
        self.assertEqual(b.grad.numpy().item(), -2.0/(4.0**2))

    def test_div_scalar(self):
        a = pg.Tensor(3.0, requires_grad=True)
        c = a / 2
        c.backward()

        self.assertEqual(c.numpy().item(), 1.5)
        self.assertEqual(a.grad.numpy().item(), 1.0 / 2)

    def test_negate(self):
        a = pg.Tensor(3.0, requires_grad=True)
        c = -a
        c.backward()

        self.assertEqual(c.numpy().item(), -3.0)
        self.assertEqual(a.grad.numpy().item(), -1.0)

    def test_log(self):
        a = pg.Tensor(2.0, requires_grad=True)
        c = pg.log(a)
        c.backward()

        self.assertEqual(c.numpy().item(), np.log(2.0))
        self.assertEqual(a.grad.numpy().item(), 1.0 / 2.0)

    def test_exp(self):
        a = pg.Tensor(2.0, requires_grad=True)
        c = pg.exp(a)
        c.backward()

        self.assertEqual(c.numpy().item(), np.exp(2.0))
        self.assertEqual(a.grad.numpy().item(), np.exp(2.0))

    def test_relu(self):
        a = pg.Tensor(-4.0, requires_grad=True)
        b = pg.Tensor(2.0, requires_grad=True)
        c = pg.relu(a)
        d = pg.relu(b)
        c.backward()
        d.backward()

        self.assertEqual(c.numpy().item(), 0.0)
        self.assertEqual(d.numpy().item(), 2.0)
        self.assertEqual(a.grad.numpy().item(), 0.0)
        self.assertEqual(b.grad.numpy().item(), 1.0)


if __name__ == "__main__":
    unittest.main()

import numpy as np
import unittest

from neuralnetwork.activations import Linear, ReLU, Tanh, Sigmoid


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.linear = Linear()
        self.input = np.array([-50., -1.5, 0., 1.5, 50.])
        self.output = self.linear(self.input)
        self.grad = self.linear.gradient(self.input)

    def test_shapes(self):
        self.assertEqual(self.input.shape, self.output.shape)
        self.assertEqual(self.input.shape, self.grad.shape)

    def test_call(self):
        self.assertAlmostEqual(self.output[0], -50.)
        self.assertAlmostEqual(self.output[1], -1.5)
        self.assertAlmostEqual(self.output[2], 0.)
        self.assertAlmostEqual(self.output[3], 1.5)
        self.assertAlmostEqual(self.output[4], 50.)

    def test_gradient(self):
        self.assertAlmostEqual(self.grad[0], 1.)
        self.assertAlmostEqual(self.grad[1], 1.)
        self.assertAlmostEqual(self.grad[2], 1.)
        self.assertAlmostEqual(self.grad[3], 1.)
        self.assertAlmostEqual(self.grad[4], 1.)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.relu = ReLU()
        self.input = np.array([-50., -1.5, 0., 1.5, 50.])
        self.output = self.relu(self.input)
        self.grad = self.relu.gradient(self.input)

    def test_shapes(self):
        self.assertEqual(self.input.shape, self.output.shape)
        self.assertEqual(self.input.shape, self.grad.shape)

    def test_call(self):
        self.assertAlmostEqual(self.output[0], 0.)
        self.assertAlmostEqual(self.output[1], 0.)
        self.assertAlmostEqual(self.output[2], 0.)
        self.assertAlmostEqual(self.output[3], 1.5)
        self.assertAlmostEqual(self.output[4], 50)

    def test_gradient(self):
        self.assertAlmostEqual(self.grad[0], 0.)
        self.assertAlmostEqual(self.grad[1], 0.)
        self.assertAlmostEqual(self.grad[2], 0.)
        self.assertAlmostEqual(self.grad[3], 1.)
        self.assertAlmostEqual(self.grad[4], 1.)


class TestTanh(unittest.TestCase):
    def setUp(self):
        self.tanh = Tanh()
        self.input = np.array([-50., -1.5, 0., 1.5, 50.])
        self.output = self.tanh(self.input)
        self.grad = self.tanh.gradient(self.input)

    def test_shapes(self):
        self.assertEqual(self.input.shape, self.output.shape)
        self.assertEqual(self.input.shape, self.grad.shape)

    def test_call(self):
        self.assertAlmostEqual(self.output[0], -1.)
        self.assertAlmostEqual(self.output[1], -0.90514825)
        self.assertAlmostEqual(self.output[2], 0.)
        self.assertAlmostEqual(self.output[3], 0.90514825)
        self.assertAlmostEqual(self.output[4], 1.)

    def test_gradient(self):
        self.assertAlmostEqual(self.grad[0], 0.)
        self.assertAlmostEqual(self.grad[1], 0.18070663)
        self.assertAlmostEqual(self.grad[2], 1)
        self.assertAlmostEqual(self.grad[3], 0.18070663)
        self.assertAlmostEqual(self.grad[4], 0.)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()
        self.input = np.array([-50., -1.5, 0., 1.5, 50.])
        self.output = self.sigmoid(self.input)
        self.grad = self.sigmoid.gradient(self.input)

    def test_shapes(self):
        self.assertEqual(self.input.shape, self.output.shape)
        self.assertEqual(self.input.shape, self.grad.shape)

    def test_call(self):
        self.assertAlmostEqual(self.output[0], 0.)
        self.assertAlmostEqual(self.output[1], 0.18242552)
        self.assertAlmostEqual(self.output[2], 0.5)
        self.assertAlmostEqual(self.output[3], 0.81757447)
        self.assertAlmostEqual(self.output[4], 1.)

    def test_gradient(self):
        self.assertAlmostEqual(self.grad[0], 0.)
        self.assertAlmostEqual(self.grad[1], 0.14914645)
        self.assertAlmostEqual(self.grad[2], 0.25)
        self.assertAlmostEqual(self.grad[3], 0.14914645)
        self.assertAlmostEqual(self.grad[4], 0.)


if __name__ == "__main__":
    unittest.main()

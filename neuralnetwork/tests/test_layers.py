import numpy as np
import unittest

from neuralnetwork.layers import Dense


class TestDense(unittest.TestCase):
    def setUp(self):
        self.input_size = 30
        self.units = 100
        self.layer = Dense(self.input_size, self.units)

    def test_forward_shape(self):
        x = np.random.rand(1, self.input_size)
        self.assertEqual(self.layer.forward_propagate(x).shape, (1, self.units))

    def test_backward_shape(self):
        x = np.random.rand(1, self.input_size)
        self.layer.forward_propagate(x)
        output_gradient = np.random.rand(1, self.units)
        learning_rate = 0.1
        self.assertEqual(self.layer.backward_propagate(output_gradient, learning_rate).shape, (1, self.input_size))


if __name__ == "__main__":
    unittest.main()

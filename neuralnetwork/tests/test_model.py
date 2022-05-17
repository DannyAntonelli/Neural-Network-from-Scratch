import numpy as np
import unittest

from neuralnetwork.model import NeuralNetwork
from neuralnetwork.layers import Dense
from neuralnetwork.losses import MSE


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.input_size = 100
        self.output_size = 50
        self.model = NeuralNetwork(
            layers=[
                Dense(input_size=self.input_size, units=500),
                Dense(input_size=500, units=300),
                Dense(input_size=300, units=self.output_size)
            ],
            loss=MSE()
        )

    def test_add(self) -> None:
        previous_layers = len(self.model._layers)
        layer = Dense(self.output_size, 10)
        self.model.add(layer)
        self.assertEqual(len(self.model._layers), previous_layers + 1)
        self.assertEqual(self.model._layers[-1], layer)

    def test_predict_shape(self) -> None:
        samples = 200
        x = np.random.rand(samples, self.input_size)
        self.assertEqual(self.model.predict(x).shape, (samples, self.output_size))


if __name__ == "__main__":
    unittest.main()

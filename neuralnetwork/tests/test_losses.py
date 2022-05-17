import numpy as np
import unittest

from neuralnetwork.losses import MSE, BinaryCrossEntropy


class TestMSE(unittest.TestCase):
    def setUp(self) -> None:
        self.mse = MSE()
        self.predictions = np.array([-34.2, 21.6, .05, 14.9, 44.])
        self.labels = np.array([-35., 22.5, 0., 15.3, 45.1])
        self.output = self.mse(self.predictions, self.labels)
        self.grad = self.mse.gradient(self.predictions, self.labels)

    def test_shapes(self) -> None:
        self.assertEqual(self.output.shape, ())
        self.assertEqual(self.grad.shape, self.predictions.shape)

    def test_call(self) -> None:
        self.assertAlmostEqual(self.output, 0.5645)

    def test_gradient(self) -> None:
        self.assertAlmostEqual(self.grad[0], 0.32)
        self.assertAlmostEqual(self.grad[1], -0.36)
        self.assertAlmostEqual(self.grad[2], 0.02)
        self.assertAlmostEqual(self.grad[3], -0.16)
        self.assertAlmostEqual(self.grad[4], -0.44)


class TestBinaryCrossEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.bce = BinaryCrossEntropy()
        self.predictions = np.array([.06, .88, .79, .2, .11])
        self.labels = np.array([0, 1, 1, 0, 0])
        self.output = self.bce(self.predictions, self.labels)
        self.grad = self.bce.gradient(self.predictions, self.labels)

    def test_shapes(self) -> None:
        self.assertEqual(self.output.shape, ())
        self.assertEqual(self.grad.shape, self.predictions.shape)

    def test_call(self) -> None:
        self.assertAlmostEqual(self.output, 0.15302169)

    def test_gradient(self) -> None:
        self.assertAlmostEqual(self.grad[0], 1.06382978)
        self.assertAlmostEqual(self.grad[1], -1.13636363)
        self.assertAlmostEqual(self.grad[2], -1.26582278)
        self.assertAlmostEqual(self.grad[3], 1.25)
        self.assertAlmostEqual(self.grad[4], 1.12359550)


if __name__ == "__main__":
    unittest.main()

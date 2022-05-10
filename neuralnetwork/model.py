import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.losses import Loss


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: Loss):
        self._layers = layers
        self._loss = loss

    def add(self, layer: Layer):
        self._layers.append(layer)

    def _forward_propagate(self, output: np.ndarray):
        for layer in self._layers:
            output = layer.forward_propagate(output)
        return output

    def _backward_propagate(self, prediction: np.ndarray, label: np.ndarray, learning_rate: float):
        gradient = self._loss.gradient(prediction, label)
        for layer in reversed(self._layers):
            gradient = layer.backward_propagate(gradient, learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array(list(map(self._forward_propagate, x))).flatten()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        return self._loss(self._forward_propagate(x), y)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, verbose: bool=False):
        for epoch in range(epochs):
            loss: float = .0

            for i in range(len(x)):
                prediction = self._forward_propagate(np.array([x[i]]))
                self._backward_propagate(prediction, np.array([y[i]]), learning_rate)
                loss += self._loss(prediction, np.array([y[i]]))

            loss /= len(x)
            if verbose:
                print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss:0.4f}')

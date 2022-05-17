from abc import ABC, abstractmethod
import numpy as np

from .activations import Activation, Linear


class Layer(ABC):
    '''
    Protocol implemented by neural network layers
    '''

    @abstractmethod
    def forward_propagate(self, input: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        ...


class Dense(Layer):
    def __init__(self, input_size: int, units: int, activation: Activation=Linear()):
        self._weights = np.random.randn(input_size, units) * np.sqrt(2. / input_size)
        self._bias = np.zeros((1, units))
        self._activation = activation

    def forward_propagate(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = np.dot(x, self._weights) + self._bias
        return self._activation(self._z)

    def backward_propagate(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        output_gradient *= self._activation.gradient(self._z)
        self._weights -= learning_rate * np.dot(self._x.T, output_gradient)
        self._bias -= learning_rate * output_gradient
        return np.dot(output_gradient, self._weights.T)

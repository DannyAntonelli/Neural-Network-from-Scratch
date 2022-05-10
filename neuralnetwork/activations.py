from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    '''
    Protocol implemented by activation function classes
    '''
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        ...


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class ReLU(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.greater(x, 0).astype(int)


class Tanh(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1 + np.exp(-1 * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    '''
    Protocol implemented by loss function classes
    '''
    @abstractmethod
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.float64:
        ...

    @abstractmethod
    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...


class MSE(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.float64:
        return np.mean(np.square(predictions - labels))

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return 2 * (predictions - labels) / predictions.size


class BinaryCrossEntropy(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.float64:
        return -np.mean(np.multiply(labels, np.log(predictions)) + np.multiply(1 - labels, np.log(1 - predictions)))

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return -np.divide(labels, predictions) + np.divide(1 - labels, 1 - predictions)

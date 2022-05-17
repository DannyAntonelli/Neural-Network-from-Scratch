import numpy as np

from neuralnetwork.model import NeuralNetwork
from neuralnetwork.layers import Dense
from neuralnetwork.activations import Tanh, Sigmoid, ReLU
from neuralnetwork.losses import MSE

from keras.datasets import mnist
from keras.utils import np_utils


INPUT_DIMENSION = 28 * 28
OUTPUT_DIMENSION = 10
TRAIN_SIZE = 2000

def modify_x(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], INPUT_DIMENSION) / 255.

def get_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = modify_x(x_train)
    y_train = np_utils.to_categorical(y_train)

    x_test = modify_x(x_test)
    y_test = np_utils.to_categorical(y_test)

    return (x_train[:TRAIN_SIZE], y_train[:TRAIN_SIZE]), (x_test, y_test)

def get_model() -> NeuralNetwork:
    return NeuralNetwork(
        layers=[
            Dense(input_size=INPUT_DIMENSION, units=100, activation=Tanh()),
            Dense(input_size=100, units=50, activation=ReLU()),
            Dense(50, OUTPUT_DIMENSION, activation=Sigmoid())
        ],
        loss=MSE()
    )

def get_max_index(x: np.ndarray) -> int:
    return np.where(x == np.amax(x))[0][0]

def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_data()

    model = get_model()
    model.fit(x_train, y_train, epochs=50, learning_rate=0.1, verbose=True)

    predictions = model.predict(x_test)[:10]

    print("\n\nPrediction on the first 10 test examples:\n")
    for a, b in zip(predictions, y_test[:10]):
        print(f"predicted: {get_max_index(a)}, actual: {get_max_index(b)}")

    print(f"\nLoss on test set: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    main()

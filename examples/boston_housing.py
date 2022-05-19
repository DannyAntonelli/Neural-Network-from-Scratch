import numpy as np

from neuralnetwork.model import NeuralNetwork
from neuralnetwork.layers import Dense
from neuralnetwork.activations import ReLU
from neuralnetwork.losses import MSE

from keras.datasets import boston_housing


INPUT_DIMENSION = 13
OUTPUT_DIMENSION = 1

def standardized(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)

def modify_y(y: np.ndarray) -> np.ndarray:
    return np.array([[elem] for elem in y])

def get_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_train = standardized(x_train)
    y_train = modify_y(y_train)

    x_test = standardized(x_test)
    y_test = modify_y(y_test)

    return (x_train, y_train), (x_test, y_test)

def get_model() -> NeuralNetwork:
    return NeuralNetwork(
        layers=[
            Dense(input_size=INPUT_DIMENSION, units=150, activation=ReLU()),
            Dense(input_size=150, units=200, activation=ReLU()),
            Dense(input_size=200, units=OUTPUT_DIMENSION)
        ],
        loss=MSE()
    )

def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_data()

    model = get_model()
    model.fit(x_train, y_train, epochs=100, learning_rate=0.0001, verbose=True)

    predictions = model.predict(x_test)[:10]

    print("\n\nPrediction on the first 10 test examples:\n")
    for a, b in zip(predictions, y_test[:10]):
        print(f"predicted: {a[0]:.1f}, actual: {b[0]}")

    print(f"\nLoss on test set: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    main()

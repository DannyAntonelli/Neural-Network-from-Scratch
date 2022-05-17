import numpy as np

from neuralnetwork.model import NeuralNetwork
from neuralnetwork.layers import Dense
from neuralnetwork.activations import Sigmoid
from neuralnetwork.losses import BinaryCrossEntropy


def get_model() -> NeuralNetwork:
    return NeuralNetwork(
        layers=[
            Dense(2, 3, Sigmoid()),
            Dense(3, 1, Sigmoid())
        ],
        loss=BinaryCrossEntropy()
    )

def main() -> None:
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    model = get_model()

    model.fit(x_train, y_train, epochs=2000, learning_rate=0.5)
    predictions = model.predict(x_train)

    for i in range(len(x_train)):
        print(f"{x_train[i][0]} ^ {x_train[i][1]}, ", end="")
        print(f"predicted: {predictions[i][0]:.4f}, actual: {y_train[i][0]}")

    print(f"\nLoss on train set: {model.evaluate(x_train, y_train):.4f}")

if __name__ == "__main__":
    main()

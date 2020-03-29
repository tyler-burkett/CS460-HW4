import numpy as np
from itertools import tee, chain


def pairwise(i):
    """
    Iterate pairwise through an iterable object.

    Link: https://stackoverflow.com/questions/5764782/iterate-through-pairs-of-items-in-a-python-list
    """
    a, b = tee(i)
    next(b, None)
    return zip(a, b)


def ReLU(x):
    return max(0, x)


def dReLU(x):
    return 1 if x > 0 else 0


class Perceptron:
    """A perceptron (a.k.a. nueral network)."""

    def __init__(self, activation_func, derivative_func, node_counts, learning_rate):
        """
        Create perceptron.

        activation_func - function to compute activation function of neurons (must return a number)
        derivative_func - function to compute derivative of activation function of neurons (must return a number)
        node_counts - list of number; each index is a layer with the value being the number of nodes in said layer
        learning_rate - hyperparameter to control how much weights change during training
        """
        assert(len(node_counts) >= 1 and callable(activation_func) and callable(derivative_func))
        self.activation_func = activation_func
        self.derivative_func = derivative_func
        self.node_counts = node_counts
        self.layers = len(node_counts)
        self.weights = [np.matrix(np.random.rand(x, y)) for x, y in pairwise(node_counts)]
        self.learning_rate = learning_rate

    def fit(self, train_data):
        """
        Train perceptron of test data.

        train_data - array of test data (arrays); first index is label
        """
        activations = list(0 for i in range(self.layers))
        deltas = list(0 for i in range(self.layers - 1))
        # Assuming each instance is a column vector, with first index as label
        for data in train_data.T:
            actual = np.array(data[0])
            column_vector = data[1:]

            # Calc activations
            activations[0] = column_vector
            for i in range(1, self.layers):
                inputs = list(chain(*np.matrix.tolist(self.weights[i-1].T * np.r_["c", activations[i-1]])))
                activations[i] = np.array(list(map(self.activation_func, inputs)))

            # Calc error
            prediction = activations[-1]
            error = actual - prediction

            # Calc deltas
            deltas[-1] = np.array([error[j] * self.derivative_func(activations[-2][j]) for j in range(len(activations[-1]))])
            for delta_level in range(-2, -(len(deltas)+1), -1):
                deltas[delta_level] = np.zeros(len(self.node_counts[delta_level]))
                for i in range(len(deltas[delta_level])):
                    result = self.derivative_func(self.weights[delta_level-1].T * np.r_["c", activations[delta_level-1]]) * \
                        sum(self.weights[delta_level].item(i, j) * deltas[delta_level+1][j] for j in range(len(deltas[delta_level+1])))
                    deltas[delta_level][i] = result

            # Perform weight adjustments
            for weights in self.weights:
                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        weights[i][j] = weights.item(i, j) + self.learning_rate * activations[i][j] * deltas[i+1][j]

    def predict(self, input):
        """
        Pass input data into the perceptron to get a prediction.

        input - numpy column array
        """
        assert(input.shape == (self.node_counts[0], 1))
        activations = list(0 for i in range(self.layers))

        # Calc activations
        activations[0] = np.matrix(input)
        for i in range(1, self.layers):
            activations[i] = np.matrix(np.r_["c", map(self.activation_func, self.weights[i-1].T * activations[i-1])])

        return activations[-1]

import numpy as np
import math
from itertools import tee, chain
from statistics import mean

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

def Sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))

def dSigmoid(x):
    return Sigmoid(x)*(1 - Sigmoid(x))

class Perceptron:
    """A perceptron (a.k.a. nueral network)."""

    def __init__(self, activation_func, derivative_func, node_counts, learning_rate, iterations):
        """
        Create perceptron.

        activation_func - function to compute activation function of neurons (must return a number)
        derivative_func - function to compute derivative of activation function of neurons (must return a number)
        node_counts - list of number; each index is a layer with the value being the number of nodes in said layer
        learning_rate - hyperparameter to control how much weights change during training
        """
        assert(len(node_counts) > 1 and callable(activation_func) and callable(derivative_func))
        self.activation_func = activation_func
        self.derivative_func = derivative_func
        self.node_counts = node_counts
        self.layers = len(node_counts)
        # Weights are x rows (input), y columns (output to next layer)
        self.weights = [np.matrix(np.random.uniform(-1, 1, (x+1, y))) for x, y in pairwise(node_counts)]
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, train_data):
        """
        Train perceptron of test data.

        train_data - array of test data (arrays); first index is label
        """
        activations = list(0 for i in range(self.layers))
        deltas = list(0 for i in range(self.layers - 1))
        learning_rate = self.learning_rate

        # Assuming each instance is a column vector, with first index as label
        prev_error = np.array([float("inf")])
        for _ in range(self.iterations):
            for data in train_data.T:
                actual = np.array(data[0])
                # Extra 1 is bias input
                column_vector = np.append(np.array([1.0]), data[1:])

                # Calc activations
                activations[0] = column_vector
                for i in range(1, self.layers):
                    # a_0 = x_0, a_i = g(w_i-1^T*x_i-1)
                    inputs = list(chain(*np.matrix.tolist(self.weights[i-1].T * np.r_["c", activations[i-1]])))
                    # Each level has bias input of one
                    activations[i] = np.append(np.array([1.0]), np.array(list(map(self.activation_func, inputs))))

                # Calc error
                prediction = activations[-1][1:]
                error = actual - prediction

                # Calc deltas
                # delta_j = Error_j * g'(in)
                deltas[-1] = np.array([error[j] * self.derivative_func(activations[-2][j+1]) for j in range(0, len(error))])
                for delta_level in range(-2, -(len(deltas)+1), -1):
                    # delta_i = g'(in) * sum(w_i,j * delta_j) forall j
                    inputs = list(chain(*np.array(self.weights[delta_level].T * np.r_["c", activations[delta_level-1]]).tolist()))
                    weighted_sums = self.weights[delta_level+1] * np.r_["c", deltas[delta_level+1]]
                    dg_in = np.r_["c", list(map(self.activation_func, inputs))]
                    deltas[delta_level] = np.multiply(dg_in, weighted_sums[1:])

                # Perform weight adjustments
                for level in range(len(self.weights)):
                    self.weights[level] = self.weights[level] + learning_rate * np.matrix(activations[level]).T * np.matrix(deltas[level]).T

                if mean(error) > mean(prev_error):
                    learning_rate = max(learning_rate / 2, 10**-10)
                else:
                    learning_rate = min(learning_rate * 2, 10**-1)

                prev_error = error

    def predict(self, input):
        """
        Pass input data into the perceptron to get a prediction.

        input - numpy column array
        """
        assert(input.shape == (self.node_counts[0], 1) or input.shape == (self.node_counts[0],))
        activations = list(0 for i in range(self.layers))

        # Calc activations
        activations[0] = np.append(np.array([1.0]), input)
        for i in range(1, self.layers):
            inputs = list(chain(*np.matrix.tolist(self.weights[i-1].T * np.r_["c", activations[i-1]])))
            activations[i] = np.append(np.array([1.0]), np.array(list(map(self.activation_func, inputs))))

        return activations[-1][1:]

    def set_weights(self, weights):
        assert all(weights[i].shape == self.weights[i].shape for i in range(len(self.weights)))
        self.weights = weights

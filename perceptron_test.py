import numpy as np
from perceptron import Perceptron, ReLU, dReLU, Sigmoid


def divide(start, divisor=2, iterations=float("inf")):
    val = start
    iters = 0
    while val > 1 and iters < iterations:
        yield val
        val = val // divisor
        iters = iters + 1


if __name__ == "__main__":

    # Import test and train data
    #train_data = np.genfromtxt("./data/mini_train_0_1.csv", dtype="float", delimiter=",").T
    #test_data = train_data
    train_data = np.genfromtxt("./data/mnist_train_0_1.csv", dtype="float", delimiter=",").T
    test_data = np.genfromtxt("./data/mnist_test_0_1.csv", dtype="float", delimiter=",").T
    node_levels = list(divide(len(train_data[1:, 0]), 4)) + [1]
    nn = Perceptron(ReLU, dReLU, node_levels, 1*10**-10, 1000*5)

    nn.fit(train_data)

    total = len(test_data[0])
    num_correct = 0
    for test_point in test_data.T:
        actual = test_point[0]
        input = test_point[1:]
        prediction = int(nn.predict(input).item(0) >= 0.5)
        print("Actual: {}, Prediction: {}".format(actual, prediction))
        if prediction == actual:
            num_correct = num_correct + 1
    print("Accuracy: {}".format(num_correct / total))
    import code
    code.interact(local=locals())

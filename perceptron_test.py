import numpy as np
from perceptron import Perceptron, Sigmoid, dSigmoid
from statistics import mean

if __name__ == "__main__":

    # Import test and train data
    train_data = np.genfromtxt("./data/mnist_train_0_1.csv", dtype="float", delimiter=",").T
    test_data = np.genfromtxt("./data/mnist_test_0_1.csv", dtype="float", delimiter=",").T

    # Normalize data
    train_data[1:, :] = train_data[1:, :] / 255
    test_data[1:, :] = test_data[1:, :] / 255

    # Construct number of neurons
    node_levels = [len(train_data[1:, 0]), len(train_data[1:, 0])//8, 1]

    best_accuracy = 0
    accuracies = list()
    best_weights = None
    for i in range(10):
        nn = Perceptron(Sigmoid, dSigmoid, node_levels, 10**-10, 10)

        print("start training")
        nn.fit(train_data)

        total = len(test_data[0])
        num_correct = 0
        for test_point in test_data.T:
            actual = test_point[0]
            input = test_point[1:]
            activation = nn.predict(input).item(0)
            prediction = float(activation > 0.5)
            if prediction == actual:
                num_correct = num_correct + 1
        current_accuracy = num_correct / total
        accuracies.append(current_accuracy)
        print("Test {} Accuracy: {}".format(i, current_accuracy))
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            weights = nn.weights

    print("Average accuracy: {}".format(mean(accuracies)))
    print("Best accuracy: {}".format(best_accuracy))
    print("Worst accuracy: {}".format(min(accuracies)))

import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def softmax(class_num, w, b, x_t):
    """
    Softmax of w returns the probability that x_t belongs to class class_num + 1.
    :param class_num: class number in w (minus 1)
    :param w: matrix
    :param b: const vector
    :param x_t: example to check
    :return: probability that x_t belongs to class class_num + 1.
    """
    normal_of_vec = 0
    for j in xrange(3):
        normal_of_vec += np.exp(w[j] * x_t + b[j])
    return np.exp(w[class_num] * x_t + b[class_num]) / normal_of_vec


def norm_dist_density(x, mean, variance):
    """
    Implementation of pdf (probability density function).
    :param x: input
    :param mean: mean.
    :param variance: variance.
    :return: output of pdf according to x.
    """
    return (1.0 / (variance * sqrt(2 * pi))) * np.exp((-(x - mean) ** 2) / (2 * (variance ** 2)))


def update_set(set):
    """
    Update the training set with 100 examples for each class (N(2,1), N(4,1), N(6,1))
    :param set: the training set.
    :return: -
    """
    variance = 1.0
    total_examples_per_class = 100
    for class_num in xrange(1, 4):
        # 100 examples for class_num with mean of 2*class_num and variance.
        examples = np.random.normal(2 * class_num, variance, total_examples_per_class)
        for generated_x in examples:
            set.append((generated_x, class_num))


def training(w, b):
    """
    Update the training set using update_set function, then run over the training set (epochs times) and
    update w and b according to the same rules of the theoretical part (logistic regression with SGD update rule)
    so we update w and b for every example.
    :param w: matrix
    :param b: const vector
    :return: -
    """
    set = []
    update_set(set)

    eta = 0.1
    epochs = 20
    for epoch in xrange(epochs):
        np.random.shuffle(set)
        for (x_t, tag) in set:
            for class_num in xrange(3):
                if class_num + 1 != tag:
                    w_t = softmax(class_num, w, b, x_t) * x_t
                    b_t = softmax(class_num, w, b, x_t)
                else:
                    w_t = -x_t + softmax(class_num, w, b, x_t) * x_t
                    b_t = -1 + softmax(class_num, w, b, x_t)
                # update of w and b according to the example and it's tag
                w[class_num] -= eta * w_t
                b[class_num] -= eta * b_t


def update_true_distribution(true_dist):
    """
    Update of the true distribution list.
    :param true_dist: the true distribution list.
    :return: -
    """
    for x in xrange(0, 11):
        true_dist[x] = (norm_dist_density(x, 2, 1) /
                        (norm_dist_density(x, 2, 1) + norm_dist_density(x, 4, 1) + norm_dist_density(x, 6, 1)))


def update_trained_distribution(trained_dist, w, b):
    """
    Update of the trained distribution list.
    :param trained_dist: calculation of the probability for every x between 1 and 10 to be in class 1.
    :param w: matrix.
    :param b: const vector
    :return: -
    """
    for x in range(0, 11):
        trained_dist[x] = softmax(0, w, b, x)


def print_graphs(w, b):
    """
    Make graphs for (x values are 1-10, y values are changes for some x to be in class 1 by the distribution):
        Create the true_distribution graph values from update_true_distribution
        Create the trained_distribution graph values from update_trained_distribution
    Plot the graphs
    :param w: matrix.
    :param b: const vector
    :return: -
    """
    true_dist = {}
    update_true_distribution(true_dist)
    trained_dist = {}
    update_trained_distribution(trained_dist, w, b)
    norm_line, = plt.plot(true_dist.keys(), true_dist.values(), "red", label='The Normal Distribution')
    trained_line, = plt.plot(trained_dist.keys(), trained_dist.values(), "black", label='The Trained Distribution')
    plt.legend(handler_map={norm_line: HandlerLine2D()})
    plt.show()


def main():
    """
    Create w and b from scratch, train and update them and print results.
    """
    w = [0, 0, 0]
    b = [0, 0, 0]
    training(w, b)
    print_graphs(w, b)


if __name__ == "__main__":
    main()

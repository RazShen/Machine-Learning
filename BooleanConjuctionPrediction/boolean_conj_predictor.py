import sys
import numpy as np


def switchValue(val):
    """
    This function gets a value, if it's 1 return 0. else, return 1.
    :param val: value 0/1.
    :return: 0/1 (read the description)
    """
    if val:
        return 0
    else:
        return 1

def writeToFile(h, d):
    """
    This function write the finished hypothesis to a file called output.txt by the wanted format.
    :param h: the hypothesis.
    :param d: the number variables.
    """
    file = open("output.txt", "w")
    output_string = ""
    for i in range(d):
        if h[2 * i] == 1:
            output_string += str("X") + str(i + 1) + ","
        elif h[2 * i + 1] == 1:
            output_string += "not(" + str("X") + str(i + 1) + ")" + ","
    output_string = output_string.strip(",")
    file.writelines(output_string)
    file.close()


def train(h, x, y, d):
    """
    This function train the all negative hypothesis by the consistency algorithm
    :param h: all negative hypothesis
    :param x: x values (row of length d in a matrix) describing the substitutions
    :param y: y values (describe the right tag of x)
    :param d: length of row in x (number of variables)
    :return: the finished trained hypothesis
    """
    # go over the set and start the training
    for example, tag in zip(x, y):
        y_hat = 1
        # calculate y_hat (substitute x values in the hypothesis)
        for i in range(d):
            if h[2 * i] == 1:
                y_hat *= int(example[i])
            elif h[2 * i + 1] == 1:
                y_hat *= switchValue(int(example[i]))
        # check if the y_hat (the prediction) matches the correct tag in the training set.
        if tag == 1 and y_hat == 0:
            for i in range(d):
                if example[i] == 1:
                    h[2 * i + 1] = 0
                else:
                    h[2 * i] = 0
    return h

def main():
    """
    The main function of the program, loads the training set. creates an hypothesis, trains it and write it into txt
    using some helping functions.
    """
    # load the training_examples
    path = sys.argv[1]
    training_examples = np.loadtxt(path)
    # get the length of a row in the matrix
    line__len = training_examples.size/len(training_examples)

    # get the X matrix of the training set (each row has d variables)
    x = training_examples[:len(training_examples), :line__len - 1]
    # get the Y vector of the training set (number of rows is the number of the examples in the training set)
    y = training_examples[:len(training_examples), line__len - 1:]
    d = line__len - 1
    h = []

    # initialize the all-negative hypothesis
    for i in range(2*d):
        h.append(1)

    # train the hypothesis
    h = train(h, x, y, d)

    # write the final hypothesis to the file
    writeToFile(h, d)


# this function calls the main
if __name__ == '__main__':
    main()
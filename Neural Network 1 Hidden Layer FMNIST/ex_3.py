import numpy as np

CLASSES = 10
EPOCHS = 40
LEARNRATE = 0.005
HIDDENSIZE = 100
INPUTSIZE = 28 * 28
PIXELVALUES = 255


def main():
    """
    Main function, trains the model and update the parameters accordingly. finally creates a test result file.
    :return: (creates test results file)
    """
    train_x = np.load("train_x.bin.npy")
    # train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    train_x, train_y = shuffle_two_matching_mat(train_x, train_y)
    train_x = train_x / PIXELVALUES
    test_x = test_x / PIXELVALUES
    validation_size = int(int(len(train_x)) * 0.2)
    validation_examples, validation_tags = train_x[-validation_size:, :], train_y[-validation_size:]
    train_examples, train_tags = train_x[: -validation_size, :], train_y[: -validation_size]
    # print("Finishing loading")
    # Possible multiplying random by 0.01
    # W1 = np.random.rand(HIDDENSIZE, INPUTSIZE) * 0.01
    # b1 = np.random.rand(HIDDENSIZE, 1) * 0.01
    # W2 = np.random.rand(CLASSES, HIDDENSIZE) * 0.01
    # b2 = np.random.rand(CLASSES, 1) * 0.01
    # better picking random from uniform values
    W1 = np.random.uniform(-0.08, 0.08, [HIDDENSIZE, INPUTSIZE])
    b1 = np.random.uniform(-0.08, 0.08, [HIDDENSIZE, 1])
    W2 = np.random.uniform(-0.08, 0.08, [CLASSES, HIDDENSIZE])
    b2 = np.random.uniform(-0.08, 0.08, [CLASSES, 1])

    params = [W1, b1, W2, b2]
    params = train(params, sigmoid, train_examples, train_tags, validation_examples, validation_tags)
    # write to test.pred file
    prediction_file = open("test.pred", 'w')
    for x in test_x:
        x = np.reshape(x, (1, INPUTSIZE))
        result, h1, z1, z2 = f_prop(params, sigmoid, x)
        max_array = result.argmax(axis=0)
        prediction_file.write(str(max_array[0]) + "\n")
    prediction_file.close()


def shuffle_two_matching_mat(mat1, match_mat2):
    """
    This method takes 2 matrixes (each one is of the same size with is number_of_rowX1) and shuffle them synchronically)
    :param mat1: first matrix
    :param match_mat2: second matrix
    :return: shuffled natrixes.
    """
    shape = np.arange(mat1.shape[0])
    np.random.shuffle(shape)
    mat1 = mat1[shape]
    mat2 = match_mat2[shape]
    return mat1, mat2


def train(params, active_function, train_x, train_y, validation_examples, validation_tags):
    """
    Train the parameters by using forward propagation to calculate the probabilities vector, then train the
    parameters using back propagation (using the probabilities vector we found) and update_weights.
    after each epoch go through the validation set
    :param params: W1, b1, W2, b2
    :param active_function: as chosen (sigmoid/ReLU,tanh and others)
    :param train_x: training examples
    :param train_y: training tags
    :param validation_examples: validation examples
    :param validation_tags: validation tags
    :return: trained parameters.
    """
    # go over the training set EPOCHS times
    for i in xrange(EPOCHS):
        sum_loss = 0.0
        train_x, train_y = shuffle_two_matching_mat(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, INPUTSIZE))
            # get y_hat from softmax(W2*(activation_function(W1*x+b1))+b2)
            classification_vec, h1, z1, z2 = f_prop(params, active_function, x)
            sum_loss += loss(classification_vec, int(y))
            gradient_mat = b_prop(params, h1, classification_vec, z1, z2, active_function, x, y)
            params = update_weights(params, gradient_mat)
        validation_set_loss, accuracy = get_validation_results(params, active_function, validation_examples,
                                                               validation_tags)
        print i, sum_loss / train_x.shape[0], validation_set_loss, accuracy * 100
    return params


def f_prop(params, active_function, x):
    """
    Do the f_prop by the recitation notes (calculate the hidden layer (from z1) and than use w2 and b2 to make an
    output and send it to softmax for a probabilities vector).
    :param params: W1, b1, W2, b2
    :param active_function: as chosen (sigmoid/ReLU,tanh and others)
    :param x: example to analyze
    :return: probabilities vector
    """
    w1, b1, w2, b2 = params
    x = np.transpose(x)
    z1 = np.dot(w1, x) + b1
    h1 = active_function(z1)
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(w2, b2, h1)
    return y_hat, h1, z1, z2


def b_prop(params, h1, h2, z1, z2, active_function, x, y):
    """
    Do the derivatives by the recitation notes, all are by the chain rule and simplified
    by the numpy matrix multiplication
    :param params: W1, b1, W2,b2
    :param h1: h1
    :param h2: h2
    :param z1: z1
    :param z2: z2
    :param active_function: as chosen (sigmoid/ReLU,tanh and others)
    :param x: train example
    :param y: train tag
    :return: gradient matrix for W1, b1, W2, b2 (to increase the right tag and decrease the others by the update rule).
    """
    w1, b1, w2, b2 = params
    h2_new = h2
    h2_new[int(y)] -= 1  # sigmoid vector - one hot vector
    grad_w2 = np.dot(h2_new, np.transpose(h1))  # dL/dW2 = dL/dZ2 * dZ2/dW2
    grad_b2 = h2_new  # dL/dB2 = dL/dZ2 * dZ2/dB2
    # dL/dZ1 = dL/dZ2 * dZ2/dH1 * dH1/dZ1
    dz1 = np.dot(np.transpose(w2), h2_new) * active_function(z1) * (1 - active_function(z1))
    grad_w1 = np.dot(dz1, x)  # dL/dW1 = dL/dZ2 * dZ2/dH1 * dH1/dZ1 * dZ1/dW1
    grad_b1 = dz1  # dL/dB1 = dL/dZ2 * dZ2/dH1 * dH1/dZ1 * dZ1/dB1
    return grad_w1, grad_b1, grad_w2, grad_b2


def loss(classification_vec, y):
    """
    calculate the loss function (which is basically -(sigma(y*log(y_hat))
    :param classification_vec: y_hat
    :param y: right tag
    :return: the loss (float)
    """
    return -np.log(classification_vec[int(y)])


def update_weights(params, gradient_mat):
    """
    Update the weights by the sgd update rule.
    :param params: W1, b1, W2, b2
    :param gradient_mat: of every parameter
    :return: updated parameters.
    """
    w1, b1, w2, b2 = params
    grad_w1, grad_b1, grad_w2, grad_b2 = gradient_mat
    w1 -= LEARNRATE * grad_w1
    w2 -= LEARNRATE * grad_w2
    b1 -= LEARNRATE * grad_b1
    b2 -= LEARNRATE * grad_b2
    return w1, b1, w2, b2


def get_validation_results(params, active_function, validation_examples, validation_tags):
    """
    Go over the validation set, use only fprop (includes the sigmoid) and calculate the loss (so that i'll know
    when the model isn't learning and is overfitting). don't update the parameters of the model.
    :param params: W1,b1,W2,b2
    :param active_function: As chosen (sigmoid/ReLU,tanh and others)
    :param validation_examples: validation examples
    :param validation_tags: validation tags
    :return: the loss and the accuracy of the validation tags.
    """
    true_positive = 0
    sum_loss = 0
    for x, y in zip(validation_examples, validation_tags):
        x = np.reshape(x, (1, INPUTSIZE))
        # get y_hat (probabilities vector)
        y_hat, h1, h2, z2 = f_prop(params, active_function, x)
        loss_val = loss(y_hat, y)
        max_array = y_hat.argmax(axis=0)
        sum_loss += loss_val
        if max_array[0] == int(y):
            true_positive += 1
    # calculate the how many examples we were right about from all the examples.
    acc = true_positive / float(len(validation_tags))
    # calculate the loss of the validation (sum of all the loss) divided by numbere of the validation examples.
    avg_loss = sum_loss / validation_examples.shape[0]
    return avg_loss, acc


def sigmoid(val):
    """
    Sigmoid function, takes a value (vector) and make all it's value be between 0 to 1.
    :param val: vector/matrix.
    :return: same vector/matrix sizes with its values limited by 1.
    """
    return np.divide(1, (1 + np.exp(-val)))


def softmax(final_w, final_b, before_last_h):
    """
    Softmax function, calculate the softmax and return 10X1 probability vec.
    :param final_w: W2
    :param final_b: b2
    :param before_last_h: h1
    :return: 10X1 probability vec.
    """
    result_vec = np.zeros((CLASSES, 1))
    sum_denominator = 0
    for j in xrange(CLASSES):
        # calculate the denominator
        sum_denominator += np.exp(np.dot(final_w[j], before_last_h) + final_b[j])
    for i in xrange(CLASSES):
        # calculate the numerator
        result_vec[i] = (np.exp(np.dot(final_w[i], before_last_h) + final_b[i])) / sum_denominator
    return result_vec


if __name__ == "__main__":
    main()

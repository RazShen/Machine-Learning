import numpy as np

CLASSES = 10
EPOCHS = 50
LEARNRATE = 0.01
HIDDENSIZE = 50
INPUTSIZE = 28*28

def main():
    train_x = np.load("train_x.bin.npy")
    # train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    train_x, train_y = shuffleTwoMatchingMat(train_x, train_y)
    train_x = train_x / 255
    test_x = test_x / 255
    validationSize = int(int(len(train_x)) * 0.2)
    validationExamples, validationTags = train_x[-validationSize:, :], train_y[-validationSize:]
    trainExamples, trainTags = train_x[: -validationSize, :], train_y[: -validationSize]
    print("Finishing loading")
    #W1 = np.random.rand(HIDDENSIZE, INPUTSIZE) * 0.01
    #b1 = np.random.rand(HIDDENSIZE, 1) * 0.01
    #W2 = np.random.rand(CLASSES, HIDDENSIZE) * 0.01
    #b2 = np.random.rand(CLASSES, 1) * 0.01
    W1 = np.random.uniform(-0.08, 0.08, [HIDDENSIZE, INPUTSIZE])
    b1 = np.random.uniform(-0.08, 0.08, [HIDDENSIZE, 1])
    W2 = np.random.uniform(-0.08, 0.08, [CLASSES, HIDDENSIZE])
    b2 = np.random.uniform(-0.08, 0.08, [CLASSES, 1])

    params = [W1, b1, W2, b2]
    params = train(params, sigmoid, trainExamples, trainTags, validationExamples, validationTags)
    predictionfile = open("test.pred", 'w')
    for x in test_x:
        x = np.reshape(x, (1, INPUTSIZE))
        result, h1, z1,z2 = f_prop(params, sigmoid, x)
        max_array = result.argmax(axis=0)
        predictionfile.write(str(max_array[0])+"\n")
    predictionfile.close()

def shuffleTwoMatchingMat(mat1, matchmat2):
    shape = np.arange(mat1.shape[0])
    np.random.shuffle(shape)
    mat1 = mat1[shape]
    mat2 = matchmat2[shape]
    return mat1, mat2


def train(params, active_function, train_x, train_y, validation_examples, validation_tags):
    for i in xrange(EPOCHS):
        sum_loss = 0.0
        train_x, train_y = shuffleTwoMatchingMat(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, INPUTSIZE))
            classification_vec, h1, z1, z2 = f_prop(params, active_function, x)
            sum_loss += loss(classification_vec, int(y))
            gradient_mat = b_prop(params, h1,classification_vec,z1, z2, active_function, x, y)
            params = update_weights(params, gradient_mat)
        validation_set_loss, accuracy = predict_on_dev(params, active_function, validation_examples, validation_tags)
        print i, sum_loss / train_x.shape[0], validation_set_loss, accuracy*100
    return params

def f_prop(params, active_function, x):
    W1, b1, W2,b2 = params
    x = np.transpose(x)
    z1 = np.dot(W1, x) + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    y_hat = softmax(W2, b2, h1)
    return y_hat, h1, z1, z2
2
def b_prop(params,h1,h2,z1, z2, active_function, x, y):
    W1, b1, W2,b2 = params
    h2_new = h2
    h2_new[int(y)] -= 1
    gradW2 = np.dot(h2_new, np.transpose(h1)) #dL/dW2
    gradB2 = h2_new #dL/dB1
    dz1 = np.dot(np.transpose(W2), h2_new) * sigmoid(z1) * (1-sigmoid(z1))
    gradW1 = np.dot(dz1, x)
    gradB1 = dz1
    return gradW1, gradB1, gradW2, gradB2


def loss(classifaction_vec, y):
    return -np.log(classifaction_vec[int(y)])

def update_weights(params, gradient_mat):
    W1, b1, W2, b2 = params
    gradW1, gradb1, gradW2, gradb2 = gradient_mat
    W1 -= LEARNRATE*gradW1
    W2 -= LEARNRATE*gradW2
    b1 -= LEARNRATE*gradb1
    b2 -= LEARNRATE*gradb2
    return W1,b1,W2,b2

def predict_on_dev(params, active_function, validationExamples, validationTags):
    true_positive = 0
    sum_loss = 0
    for x,y in zip(validationExamples, validationTags):
        x = np.reshape(x, (1, INPUTSIZE))
        y_hat,h1,h2,z2 = f_prop(params, active_function, x)
        loss_val = loss(y_hat, y)
        max_array = y_hat.argmax(axis=0)
        sum_loss += loss_val
        if max_array[0] == int(y):
            true_positive += 1
    acc = true_positive / float(len(validationTags))
    avg_loss = sum_loss / validationExamples.shape[0]
    return avg_loss, acc

def sigmoid(val):
   return np.divide(1, (1 + np.exp(-val)))

def softmax(final_w, final_b, before_last_h):

    result_vec = np.zeros((CLASSES, 1))
    dominator = 0
    for j in xrange(CLASSES):
        dominator += np.exp(np.dot(final_w[j], before_last_h) + final_b[j])
    for i in xrange(CLASSES):
        result_vec[i] = (np.exp(np.dot(final_w[i], before_last_h) + final_b[i])) / dominator
    return result_vec

    """
    val = np.dot(final_w, before_last_h) + final_b
    val -= np.max(val)
    val = np.exp(val)
    val /= np.sum(val)
    return val
    
    val = np.dot(final_w, before_last_h) + final_b
    return (np.exp(val - np.amax(val)) / (np.sum(np.exp(val - np.amax(val)))))
    """

if __name__ == "__main__":
    main()

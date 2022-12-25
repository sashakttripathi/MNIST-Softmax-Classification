import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist_data = fetch_openml('mnist_784', version=1)
X = mnist_data['data']      # pandas core frame dataframe
y = mnist_data['target']    # pandas core series series

X = X.to_numpy()
y = y.to_numpy()

X = np.c_[np.ones([X.shape[0], 1]), X]
X = X / 255.
# print(X[0])


def target_categories_to_numbers(y_):
    y_numbers = np.zeros(y_.shape[0])
    for i in range(10):
        y_numbers[np.where(y_ == np.unique(y_)[i])] = i

    return y_numbers.astype(int)


def numbers_to_one_zero_encoding(y_):
    y_one_zero_matrix = np.zeros((y_.shape[0], 10))
    y_one_zero_matrix[np.arange(y_.shape[0]), y_] = 1

    return y_one_zero_matrix


numbers_y = target_categories_to_numbers(y)
one_zero_matrix_y = numbers_to_one_zero_encoding(numbers_y)

X_training, X_testing, y_training, y_testing = train_test_split(X, one_zero_matrix_y, test_size=0.2, random_state=14)


def linear_predictor(w, x):     # W is a 785x10 matrix  # X is a number_of_values x 785 matrix
    return np.matmul(x, w)      # number_of_values x 10


def softmax_predictor(linear_value):        # linear value is a number_of_value x 10 matrix
    # print(linear_value[0])
    exponent_matrix = np.exp(linear_value)  # matrix of the same size as linear_value

    sum_of_exponential_values = np.sum(exponent_matrix, axis=1, keepdims=True)
    # matrix where each of the rows have been reduced to a single column
    softmax_prediction_ = exponent_matrix / sum_of_exponential_values
    return softmax_prediction_
    # each row is divided by the same sum value irrespective of column


def get_output_from_softmax_predictor(softmax_prediction_):  # softmax prediction is a number_of_value x 10 matrix
    return np.argmax(softmax_prediction_, axis=1)


def weight_update(softmax_prediction_, w, y_, x, alpha):     # calculated likelihood maximization through gradient
    diff = y_ - softmax_prediction_                          # ascent
    gradient = np.matmul(x.T, diff)         # diff is num_of_value x 10 matrix, x.T is 785 x num_of_value matrix
    w = w + alpha * gradient                # hence gradient is 785 x 10 matrix which is indeed also the size of
    # print(w[0])                             # our weight matrix
    return w


def cost_function(softmax_prediction_, y_):
    cost_ = -np.mean(np.sum(y_ * np.log(softmax_prediction_ + 1e-6), axis=1))
    return cost_


learning_rate = 0.0001

np.random.seed(14)
W = np.random.randn(785, 10)

epochs = 15000
batch_size = 7000

for epoch in range(epochs):
    # for i in range(0, X_training.shape[0], batch_size):
    #     X_batch = X_training[i:i + batch_size]
    #     y_batch = one_zero_matrix_y[i:i + batch_size]
    X_batch = X_training[:batch_size]
    y_batch = y_training[:batch_size]

    linear_prediction = linear_predictor(W, X_batch)
    # print(linear_prediction[0])
    softmax_prediction = softmax_predictor(linear_prediction)

    if epoch % 100 == 0:
        cost = cost_function(softmax_prediction, y_batch)
        print(epoch, cost, sep=" : ")

    W = weight_update(softmax_prediction, W, y_batch, X_batch, learning_rate)


def test_model(x_test, y_test, w):
    linear_test_prediction = linear_predictor(w, x_test)
    softmax_test_prediction = softmax_predictor(linear_test_prediction)
    final_prediction = get_output_from_softmax_predictor(softmax_test_prediction)

    accuracy = np.mean(numbers_to_one_zero_encoding(final_prediction) == y_test)
    return accuracy


print(f'Accuracy : {test_model(X_testing, y_testing, W)}')

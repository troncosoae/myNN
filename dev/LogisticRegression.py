import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def J(yhat, y):
    return sum(0.5*(y - yhat)**2)


def y_estim(x_i, w_hat, b_hat):
    return sigmoid(np.dot(w_hat, x_i) + b_hat)


def y_estim(x_i, w_hat, b_hat):
    return sigmoid(np.dot(w_hat, x_i) + b_hat)


def y_estim_all(x_data, w_hat, b_hat):
    y_estim = np.zeros(shape=(100, 1))
    for i in range(x_data.shape[0]):
        i_data = x_data[i]
        y_estim[i] = sigmoid(np.dot(i_data, w_hat) + b_hat)
    return y_estim


def log_regression(x_data, y_data, **kwargs):
    N = x_data.shape[0]
    w = kwargs.get('w0', np.zeros(shape=(x_data.shape[1], 1)))
    b = kwargs.get('b0', 0)
    alpha = kwargs.get('alpha', 1)
    epsilon = kwargs.get('epsilon', 1e-6)

    # w_hist = np.arr
    # b_hist = []
    # w_hist.append(w.flatten())
    # b_hist.append(b.flatten())

    y_hat = y_estim_all(x_data, w, b)
    err = np.sum((y_hat - y_data)**2)/N
    prev_err = 1000
    delta_err = prev_err - err

    while delta_err > epsilon or delta_err < 0:

        prev_err = err

        dw = np.sum((y_hat-y_data)*y_hat*(np.ones(shape=y_data.shape) - y_hat)*x_data, axis=0)/N
        db = np.sum((y_hat-y_data)*y_hat*(np.ones(shape=y_data.shape) - y_hat), axis=0)/N

        dw = np.array([[dw[0]], [dw[1]]])
        db = np.array([db])

        # print(dw, w, db, b)

        w -= alpha*dw
        b -= alpha*db

        # w_hist.append(w)
        # b_hist.append(b)

        y_hat = y_estim_all(x_data, w, b)
        err = np.sum((y_hat - y_data)**2)/N
        delta_err = prev_err - err

    return w, b

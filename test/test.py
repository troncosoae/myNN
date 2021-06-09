import unittest
import numpy as np
from numpy.core.fromnumeric import shape
import numpy.testing as npt

from LogisticRegression import sigmoid, J, y_estim, log_regression, y_estim_all
from plotTools import plot_features
from NeuralNetwork import NeuralNetwork


class TestNode(unittest.TestCase):

    def test_a(self):
        self.assertEqual(1, 1, "should be 1")

    def test_sigmoid(self):
        npt.assert_almost_equal(
            np.array([0.9525741268224331, 4.539786870243442e-05, 0.5]),
            sigmoid(np.array([3, -10, 0]))
        )

    def test_J(self):
        npt.assert_equal(
            17,
            J(np.array([1, 3]), np.array([4, -2]))
        )

    def test_y_estim(self):
        npt.assert_almost_equal(
            0.549833997312478,
            y_estim(np.array([1, 2, 3]), np.array([4, 1, -3]), 3.2)
        )


if __name__ == '__main__':

    # w = np.array([2, 1])
    # b = -15
    # x_data = 10*np.random.rand(100, 2)
    # y_data = np.zeros(shape=(100, 1))
    # for i in range(x_data.shape[0]):
    #     i_data = x_data[i]
    #     if np.dot(i_data, w) + b > 0:
    #         y_data[i][0] = 1

    # w_hat, b_hat = log_regression(x_data, y_data)
    # print('w_hat: ', w_hat)
    # print('b_hat: ', b_hat)

    # y_hat = np.zeros(shape=(100, 1))

    # for i in range(x_data.shape[0]):
    #     i_data = x_data[i]
    #     if sigmoid(np.dot(i_data, w_hat) + b_hat) > 0.5:
    #         y_hat[i][0] = 1

    # print('error: ', np.sum((y_hat - y_data)**2))

    nn = NeuralNetwork(alpha=1, nh=(3, 5, 2), tmax=10)
    print(nn)

    print(nn.forward_propagation(np.array([[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]])))

    print(nn.backward_propagation(np.array([[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]])))

    # unittest.main()

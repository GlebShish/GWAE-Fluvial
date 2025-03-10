import numpy as np
from core import manifolds
from sklearn.metrics import pairwise_distances


# Note: local diagonal PCA with projection
# This is the classical local diagonal PCA metric
class LocalDiagPCA:

    def __init__(self, data, sigma, rho, with_projection=False, A=None, b=None):
        self.with_projection = with_projection
        if with_projection:
            self.data = (data - b.reshape(1, -1)) @ A  # NxD
            self.A = A
            self.b = b.reshape(-1, 1)  # D x 1
        else:
            self.data = data  # NxD
        self.sigma = sigma
        self.rho = rho

    @staticmethod
    def is_diagonal():
        return True

    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        # c is D x N
        if self.with_projection:
            c = ((c.T - self.b.T) @ self.A).T

        sigma2 = self.sigma ** 2
        D, N = c.shape

        M = np.empty((N, D))
        M[:] = np.nan
        if nargout == 2:  # compute the derivative of the metric
            dMdc = np.empty((N, D, D))
            dMdc[:] = np.nan

        # TODO: replace the for-loop with tensor operations if possible.
        for n in range(N):
            cn = c[:, n]  # Dx1
            delta = self.data - cn.transpose()  # N x D
            delta2 = delta ** 2  # pointwise square
            dist2 = np.sum(delta2, axis=1, keepdims=True)  # Nx1, ||X-c||^2
            # wn = np.exp(-0.5 * dist2 / sigma2) / ((2 * np.pi * sigma2) ** (D / 2))  # Nx1
            wn = np.exp(-0.5 * dist2 / sigma2)
            print(dist2.shape)
            print(sigma2.shape)
            s = np.dot(delta2.transpose(), wn) + self.rho  # Dx1
            m = 1 / s  # D x1
            M[n, :] = m.transpose()

            if nargout == 2:
                dsdc = 2 * np.diag(np.squeeze(np.matmul(delta.transpose(), wn)))
                weighted_delta = (wn / sigma2) * delta
                dsdc = dsdc - np.matmul(weighted_delta.transpose(), delta2)
                dMdc[n, :, :] = dsdc.transpose() * m ** 2  # The dMdc[n, D, d] = dMdc_d

        if nargout == 1:
            return M
        elif nargout == 2:
            return M, dMdc




def linear_fun(x):
    return x


def d_linear_fun(x):
    return np.ones(x.shape)


def dd_linear_fun(x):
    return np.zeros(x.shape)


def softplus_fun(x):
    return np.log(1 + np.exp(x))


def d_softplus_fun(x):
    return 1 / (1 + np.exp(-x))


def dd_softplus_fun(x):
    return np.exp(x) / ((1 + np.exp(x)) ** 2)


def tanh_fun(x):
    return np.tanh(x)


def d_tanh_fun(x):
    return 1 - np.tanh(x) ** 2


def dd_tanh_fun(x):
    return -2 * np.tanh(x) * (1 - np.tanh(x) ** 2)


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid_fun(x):
    return sigmoid_fun(x) * (1 - sigmoid_fun(x))


def dd_sigmoid_fun(x):
    return d_sigmoid_fun(x) * (1 - 2 * sigmoid_fun(x))


############################################################
# Read the hidden layer and output layer activation functions
def activation_of_hidden_layer(genParams):
    if genParams['activation_fun_hidden'] == 'tanh':
        return tanh_fun, d_tanh_fun, dd_tanh_fun
    elif genParams['activation_fun_hidden'] == 'sigmoid':
        return sigmoid_fun, d_sigmoid_fun, dd_sigmoid_fun
    elif genParams['activation_fun_hidden'] == 'softplus':
        return softplus_fun, d_softplus_fun, dd_softplus_fun
    else:
        print('No valid activation function for hidden layer has been specified!')


def activation_of_output_layer(genParams):
    if genParams['activation_fun_output'] == 'tanh':
        return tanh_fun, d_tanh_fun, dd_tanh_fun
    elif genParams['activation_fun_output'] == 'sigmoid':
        return sigmoid_fun, d_sigmoid_fun, dd_sigmoid_fun
    elif genParams['activation_fun_output'] == 'softplus':
        return softplus_fun, d_softplus_fun, dd_softplus_fun
    elif genParams['activation_fun_output'] == 'linear':
        return linear_fun, d_linear_fun, dd_linear_fun
    else:
        print('No valid activation function for output layer has been specified!')

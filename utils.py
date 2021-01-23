import numpy as np
# from numba import jit

# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# @jit(nopython=True)
def sigmoid_inv(x):
    return np.log(x / (1-x))


def logsumexp(prob_lst):
    return np.exp(np.sum(np.log(prob_lst)))

# @jit(nopython=True)
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)

    return x / np.sum(x)
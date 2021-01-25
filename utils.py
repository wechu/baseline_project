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

def softmax_entropy(x):
    # x are the logits
    p = softmax(x)
    c = np.max(x)
    logpi = x - np.log(np.sum(np.exp(x - c))) - c
    return -np.sum(p * logpi)
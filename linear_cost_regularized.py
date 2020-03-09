import numpy as np


def linear_cost_regularized(X, y, theta,lamb):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    theta = theta ** 2
    return ((sq.sum() / (2 * m))+((lamb*sq.theta.sum())/(2*m)))

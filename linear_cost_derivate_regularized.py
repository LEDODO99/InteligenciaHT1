import numpy as np


def linear_cost_derivate_regularized(X, y, theta):
    h = np.matmul(X, theta)
    m, _ = X.shape
    return (np.matmul((h - y).T, X).T / m)+((lamb*sq.theta.sum())/(m))
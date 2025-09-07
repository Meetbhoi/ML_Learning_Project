import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def zscore_normalize_features(X):
    """
    Standardize dataset using Z-score normalization.
    Args:
        X (ndarray): shape (m, n)
    Returns:
        X_norm, mu, sigma
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost_logistic(X, Y, w, b):
    """
    Compute logistic regression cost.
    Args:
        X (ndarray): shape (m, n)
        Y (ndarray): shape (m, 1)
        w (ndarray): shape (n,)
        b (float)
    Returns:
        cost (float)
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -Y[i] * np.log(f_wb_i) - (1 - Y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

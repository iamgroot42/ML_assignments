import numpy as np


def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


def kernel_function(x):
	return x


def MSE(y, y_cap, theta, lamb = 0):
	diff_vec = y - y_cap
	error = np.linalg.norm(diff_vec)
	if lamb:
		theta_norm = np.linalg.norm(theta)
		error += theta * theta_norm * theta_norm
	return error / (y.shape[0])


def gradient(X, theta, y, lamb):
	n = X.shape[0]
	m = X.shape[1]
	X_tr = np.transpose(X)
	return (2.0 * ( (-X_tr * y) + ((X_tr * X + lamb * np.identity(m)) * theta) )) / n

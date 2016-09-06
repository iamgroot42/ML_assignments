import numpy as np


def MSE(y, y_cap, theta, initial_del = 0):
	diff_vec = y - y_cap
	error = np.linalg.norm(diff_vec)
	if initial_del:
		theta_norm = np.linalg.norm(theta)
		error += theta * theta_norm * theta_norm
	return error / (y.shape[0])


def gradient(X, theta, y, initial_del):
	n = X.shape[0]
	m = X.shape[1]
	X_tr = np.transpose(X)
	return (2.0 * ( (-X_tr * y) + ((X_tr * X + initial_del * np.identity(m)) * theta) )) / n

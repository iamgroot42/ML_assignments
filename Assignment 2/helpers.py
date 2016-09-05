import numpy as np


def sigmoid(x):
	return 1.0/(1 + np.exp(-x))


def kernel_function(x):
	return x


def MSE(y, y_cap, theta, lamb = 0):
	diff_vec = y - y_cap
	error = np.sum(np.transpose(diff_vec) * (diff_vec))
	if lamb:
		norm = np.linalg.norm(theta)
		error += theta * norm * norm
	return error

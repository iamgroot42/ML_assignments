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


def get_X_Y(data):
	points = []
	labels = []
	for zeta in data:
		points.append(zeta[:-1][:])
		labels.append(zeta[-1:])
	return np.matrix(points).astype(float), np.matrix(labels).astype(float)


def linear_kernel(X):
	Z = []
	for y in X:
		Z.append([1.0, y.item(0)])
	return np.matrix(Z)


def poly_kernel(X):
	d = 5
	Z = []
	print X.shape
	for y in X:
		base = 1
		temp = []
		for i in range(d):
			temp.append(base)
			base *= y.item(0)
		Z.append(temp[:])
	return np.matrix(Z)


def rbf_kernel(X):
	d = 3
	return X

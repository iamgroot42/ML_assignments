import numpy as np


def MSE(y, y_cap, theta, delta = 0):
	diff_vec = y - y_cap
	error = np.linalg.norm(diff_vec) ** 2
	theta_norm = np.linalg.norm(theta)
	error += delta * (theta_norm ** 2)
	return error / (y.shape[0])


def gradient(X, theta, y, delta):
	n = X.shape[0]
	m = X.shape[1]
	X_tr = np.transpose(X)
	return (2.0 * ( (-X_tr * y) + (((X_tr * X) + (delta * np.identity(m))) * theta) )) / n


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
	d = 2 # Hyper-parameter
	Z = []
	for y in X:
		base = 1
		temp = []
		for i in range(d+1):
			temp.append(base)
			base *= y.item(0)
		Z.append(temp[:])
	return np.matrix(Z)


def rbf_kernel(X):
	sigma = 1.0 # Hyper-parameter
	Z = []
	mu = X.mean()
	for y in X:
		Z.append([1.0, np.exp(-0.5 * (((y.item(0) - mu)/sigma) ** 2)) ])
	return np.matrix(Z)

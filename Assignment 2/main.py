import linear_regression
import matplotlib.pyplot as plt
import sys
import numpy as np
import helpers


def linear_kernel(X):
	Z = []
	for y in X:
		Z.append([1.0, y.item(0)])
	return np.matrix(Z)


def poly_kernel(X, d):
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


def rbf_kernel(X, d):
	return X


def read_file(file, add_bias = True):
	f = open(file,'r')
	x_arr = []
	y_arr = []
	for x in f:
		zeta = x.rstrip().split(',')
		x_arr.append(zeta[:-1][:])
		y_arr.append(zeta[-1:])
	X = np.matrix(x_arr).astype(float)
	Y = np.matrix(y_arr).astype(float)
	# Uncomment for linear kernel
	kernel_X = linear_kernel(X)
	# Uncomment for polynomial kernel
	# kernel_X = poly_kernel(X, 1)
	# Uncomment for gaussian kernel
	# kernel_X = rbf_kernel(X, 5)
	return kernel_X, Y


def plot_graph(start_percentage, end_percentage, step_size, X, y, iters):
	plot_data_x = []
	plot_data_y = []
	while start_percentage <= end_percentage:
		plot_data_x.append(start_percentage)
		ratio = int(len(X) * (start_percentage) / 100.0)
		error = linear_regression.linear_regression(X[:ratio], Y[:ratio], iters)
		plot_data_y.append(error[1])
		start_percentage += step_size
	plt.plot(plot_data_x, plot_data_y)
	plt.title('Data ratio v/s MSE curve')
	plt.show()


def k_fold_cross_validation(X, y, k, iters, initial_del):
	errors = []
	start = 0
	eff_k = int((k * len(X))/100.0)
	for i in range(k):
		left_data = X[:start]
		left_labels = y[:start]
		right_data = X[start + eff_k:]
		right_labels = y[start + eff_k:]
		training_data = np.append(left_data, right_data, 0)
		training_labels = np.append(left_labels, right_labels, 0)
		validation_data = X[start:start + eff_k]
		validation_labels = y[start:start + eff_k]
		theta = linear_regression.linear_regression(training_data, training_labels, iters)[0]
		errors.append(helpers.MSE(validation_labels, validation_data * theta, theta, initial_del))
		start += eff_k
	return errors


def visualize_2d_points(X, y, iters):
	theta = linear_regression.linear_regression(X, y, iters)[0]
	y_cap = X * theta
	plt.plot(X[:,1], y_cap)
	plt.scatter(X[:,1], y, c = 'green')
	plt.title('Actual points & Line produced by optimizing Theta')
	plt.show()


if __name__ == "__main__":
	fname = sys.argv[1]
	iterations = int(sys.argv[2])
	initial_del = float(sys.argv[3])
	X,Y = read_file(fname)
	# Plot data v/s MSE graph
	# plot_graph(50,90,10,X,Y,iterations)
	# Print mean, variance of erors for 10-fold cross validation
	print k_fold_cross_validation(X,Y,10,iterations,initial_del)
	# Visualize fitting the points using the trained model
	# visualize_2d_points(X, Y, iterations)

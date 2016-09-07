import linear_regression
import matplotlib.pyplot as plt
import sys
import numpy as np
import helpers


def read_file(file):
	f = open(file,'r')
	rows = []
	for x in f:
		zeta = x.rstrip().split(',')
		rows.append(zeta)
	return rows


def plot_graph(start_percentage, end_percentage, step_size, data, phi, iters, delta):
	raw_X,y = helpers.get_X_Y(data)
	X = phi(raw_X)
	plot_data_x = []
	plot_data_y = []
	while start_percentage <= end_percentage:
		plot_data_x.append(start_percentage)
		ratio = int(len(data) * (start_percentage) / 100.0)
		theta = linear_regression.linear_regression(data[:ratio], phi, iters)[0]
		plot_data_y.append(helpers.MSE(y, X * theta, theta, delta))
		start_percentage += step_size
	plt.plot(plot_data_x, plot_data_y)
	plt.title('Data ratio v/s MSE curve')
	plt.show()


def k_fold_cross_validation(data, phi, k, iters, delta):
	raw_X,y = helpers.get_X_Y(data)
	X = phi(raw_X)
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
		theta = linear_regression.linear_regression(data, phi, iters)[0]
		errors.append(helpers.MSE(validation_labels, validation_data * theta, theta, delta))
		start += eff_k
	return errors


def visualize_2d_points(data, phi, iters):
	theta = linear_regression.linear_regression(data, phi, iters)[0]
	raw_X, y = helpers.get_X_Y(data)
	X = phi(raw_X)
	y_cap = X * theta
	plt.plot(X[:,1], y_cap)
	plt.scatter(X[:,1], y, c = 'green')
	plt.title('Actual points & Line produced by optimizing Theta')
	plt.show()


if __name__ == "__main__":
	fname = sys.argv[1]
	iterations = int(sys.argv[2])
	delta = float(sys.argv[3])
	DATA = read_file(fname)
	phi = helpers.linear_kernel
	# Plot data v/s MSE graph
	# plot_graph(50, 90, 10, DATA, phi, iterations, delta)
	# Print mean, variance of erors for 10-fold cross validation
	# print k_fold_cross_validation(DATA, phi ,10,iterations,delta)
	# Visualize fitting the points using the trained model
	# visualize_2d_points(DATA, phi, iterations)

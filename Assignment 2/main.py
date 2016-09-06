import linear_regression
import matplotlib.pyplot as plt
import sys
import numpy as np


def read_file(file, add_bias = True):
	f = open(file,'r')
	x_arr = []
	y_arr = []
	for x in f:
		zeta = x.rstrip().split(',')
		data_row = zeta[:-1]
		# Add '1' for intercept term
		if(add_bias):
			data_row.insert(0,1) 
		x_arr.append(data_row[:])
		y_arr.append(zeta[-1:])
	X = np.matrix(x_arr).astype(float)
	Y = np.matrix(y_arr).astype(float)
	return X,Y


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


def k_fold_cross_validation(X, y, k, iters):
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
		errors.append(linear_regression.linear_regression(training_data, training_labels, iters)[1])
		start += eff_k
	return errors


def visualize_points(X, y, iters):
	# Visualize 'what'? :|
	theta = linear_regression.linear_regression(X, y, iters)[0]
	y_cap = X * theta
	plt.scatter(y, y_cap, c = ['red','blue'])
	plt.title('Actual points v/s Points after plotting')
	plt.show()


if __name__ == "__main__":
	fname = sys.argv[1]
	iterations = int(sys.argv[2])
	X,Y = read_file(fname)
	# Plot data v/s MSE graph
	plot_graph(50,90,10,X,Y,iterations)
	# Print mean, variance of erors for 10-fold cross validation
	print k_fold_cross_validation(X,Y,10,iterations)
	# Visualize fitting the points using the trained model
	# visualize_points(X, Y, iterations)

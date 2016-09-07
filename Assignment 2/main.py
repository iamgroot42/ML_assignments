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
		theta = linear_regression.linear_regression(data[:ratio], phi, iters, delta)
		plot_data_y.append(helpers.MSE(y, X * theta, theta, delta))
		start_percentage += step_size
	return plot_data_x, plot_data_y


def k_fold_cross_validation(data, phi, k, iters, delta):
	raw_X,y = helpers.get_X_Y(data)
	X = phi(raw_X)
	errors = []
	start = 0
	eff_k = int((k * len(X))/100.0)
	for i in range(k):
		left_data = data[:start]
		right_data = data[start + eff_k:]
		modified_data = left_data
		modified_data.extend(right_data)
		validation_data = X[start:start + eff_k]
		validation_labels = y[start:start + eff_k]
		theta = linear_regression.linear_regression(modified_data, phi, iters, delta)
		errors.append(helpers.MSE(validation_labels, validation_data * theta, theta, delta))
		start += eff_k
	return [np.mean(errors), np.var(errors)]


def visualize_2d_points(data, phi, iters, delta):
	theta = linear_regression.linear_regression(data, phi, iters, delta)
	raw_X, y = helpers.get_X_Y(data)
	X = phi(raw_X)
	y_cap = X * theta
	plt.scatter(X[:,1], y_cap, c = 'red')
	plt.scatter(X[:,1], y, c = 'green')
	plt.title('Actual points & Points produced by optimizing Theta')
	plt.show()


if __name__ == "__main__":
	fname = sys.argv[1]
	iterations = int(sys.argv[2])
	delta = float(sys.argv[3])
	DATA = read_file(fname)
	phi = helpers.poly_kernel
	# phi = helpers.linear_kernel
	# Plot data v/s MSE graph
	A,B = plot_graph(50, 90, 10, DATA, phi, iterations, delta)
	plt.plot(A, B)
	plt.title('Data ratio v/s MSE curve for ' + fname + ' (ridge-2 kernel, delta=' + str(delta) +  ')')
	plt.show()
	# Print mean, variance of erors for 10-fold cross validation
	mean, variance = k_fold_cross_validation(DATA, phi, 10, iterations, delta)
	print "Mean error(over 10-fold cross validation):", mean
	print "Mean variance(over 10-fold cross validation):", variance
	# Visualize fitting the points using the trained model
	# visualize_2d_points(DATA, phi, iterations, delta)


# Linear
# For lin : 1e-2 (learning rate)
# For sph : 1e-4 (learning rate)

# Poly(2)
# For lin : 1e-6 (learning rate)
# For sph : 1e-9 (learning rate)

# Poly(4)
# For lin : 1e-11 (learning rate)
# For sph : 1e-17 (learning rate)

# Ridge-regression (on poly2)
# For lin: 1000 (delta)
# For sph: 1000 (delta)

# 10-cross validation (2-degree polynomial)
# For sph : 
	# Mean error(over 10-fold cross validation): 0.898139576393
	# Mean variance(over 10-fold cross validation): 0.00684527720511
# For lin :
	# Mean error(over 10-fold cross validation): 1.34054042398
	# Mean variance(over 10-fold cross validation): 0.409364359781

# 10-cross validation (10000 iterations)
# For iris (learning rate = 1e-18,degree=10, delta has negligible effect unless very big)
	# Mean error(over 10-fold cross validation): 0.259285737327
	# Mean variance(over 10-fold cross validation): 0.00559854378787
# For seeds (learning rate = 1e-17,degree=6, delta has negligible effect unless very big)
	# Mean error(over 10-fold cross validation): 0.323402519063
	# Mean variance(over 10-fold cross validation): 0.0448930435135

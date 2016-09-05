import numpy
import helpers


def linear_regression(X, phi, max_itr, initial_del):
	n = X.shape[0]
	# Learning rate (hyperparameter)
	learning_rate = 0.01
	# Randomly initialized weights
	theta = np.random.rand(X.shape[1], 1)
	for i in range(max_itr):
		theta = theta - learning_rate * gradient(X, theta, phi, initial_del)
		print MSE(phi, np.transpose(theta)*X, theta, initial_del)
	return [final_parameters, J, evaluationMatrix]

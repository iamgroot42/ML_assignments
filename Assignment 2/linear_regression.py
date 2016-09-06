import numpy as np
import helpers


def linear_regression(X, phi, max_itr, initial_del = 0):
	n = X.shape[0]
	# Learning rate (hyperparameter)
	learning_rate = 1e-4
	# Randomly initialized weights
	final_parameters = np.random.rand(X.shape[1], 1)
	for i in range(max_itr):
		# print learning_rate * helpers.gradient(X, final_parameters, phi, initial_del)
		final_parameters = final_parameters - learning_rate * helpers.gradient(X, final_parameters, phi, initial_del)
		# print helpers.MSE(phi, X * final_parameters, final_parameters, initial_del)
	J = helpers.MSE(phi, X * final_parameters, final_parameters, initial_del)
	# Evaluation matrix; contains various metrics
	evaluationMatrix = []
	return [final_parameters, J, evaluationMatrix]

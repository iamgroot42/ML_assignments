import numpy as np
import helpers


def linear_regression(X, phi, max_itr, delta = 0):
	RAW_POINTS, LABELS = helpers.get_X_Y(X)
	# Apply kernel function of input data
	POINTS = phi(RAW_POINTS)
	n = POINTS.shape[0]
	# Learning rate (hyperparameter)
	learning_rate = 1e-5
	# Randomly initialized weights
	final_parameters = np.random.rand(POINTS.shape[1], 1)
	for i in range(max_itr):
		# print learning_rate * helpers.gradient(LABELS, final_parameters, phi, delta)
		final_parameters = final_parameters - learning_rate * helpers.gradient(POINTS, final_parameters, LABELS, delta)
		# print helpers.MSE(phi, LABELS * final_parameters, final_parameters, delta)
	J = helpers.MSE(LABELS, POINTS * final_parameters, final_parameters, delta)
	# Evaluation matrix; contains various metrics
	evaluationMatrix = []
	return [final_parameters, J, evaluationMatrix]

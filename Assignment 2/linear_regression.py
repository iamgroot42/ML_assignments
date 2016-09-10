import numpy as np
import helpers


def linear_regression(X, phi, max_itr, delta):
	RAW_POINTS, LABELS = helpers.get_X_Y(X)
	POINTS = phi(RAW_POINTS) # Apply kernel function of input data
	n = POINTS.shape[0]
	learning_rate = 1e-3 # Learning rate (hyperparameter)
	final_parameters = np.zeros((POINTS.shape[1], 1))
	for i in range(max_itr):
		final_parameters = final_parameters - learning_rate * helpers.gradient(POINTS, final_parameters, LABELS, delta)
	return final_parameters

# import sklearn
import sys


def sq_norm(x):
	x_norm = 0
	for entry in x:
		x_norm += entry ** 2
	return x_norm


def sub_vec(x,y):
	subtracted_vector = []
	x_len = len(x)
	for i in range(x_len):
		subtracted_vector.append(x[i] - y[i])
	return subtracted_vector


def fix_u_solve_r(indicator, X, centroids):
	K = len(centroids)
	N = len(X)
	for i in range(N):
		arg_min = 0
		for j in range(K):
			if sq_norm(sub_vec(X[i],centroids[j])) < sq_norm(sub_vec(X[i],centroids[arg_min])):
				arg_min = j
		for j in range(K):
			if j == arg_min:
				indicator[j][i] = 1
			else:
				indicator[j][i] = 0


def fix_r_solve_u(indicator, X, centroids):
	for centroid in range(len(centroids)):
		numerator = 0
		denominator = float(sum(indicator[centroid]))
		N = len(indicator[centroid])
		feature_space = len(X[0])
		numerator = []
		for i in range(feature_space):
			temp_numerator = 0
			for j in range(N):
				temp_numerator += indicator[centroid][j] * X[j][i]
			numerator.append(temp_numerator/denominator)
		centroids[centroid] = numerator


def my_kMeans(X, initial_centroids, max_iters):
	"""Runs k-means for given data and computes some metrics.
	Input parameters:
	X:- Data set matrix where each row of X is represents a single training example.
	initial_centroids:- Matrix storing initial centroid position.
	max_iters:- Maximum number of iterations that K means should run for.

	Return values:
	newCentroid:- Matrix storing final cluster centroids.
	evaluationMatrix:- Array that returns MI, AMI, RI, ARI.
	"""
	K = len(initial_centroids)
	N = len(X)
	indicator = []
	# Create a K*N matrix
	for i in range(K):
		inner_temp = []
		for j in range(N):
			inner_temp.append(0)
		indicator.append(inner_temp)
	newCentroid = initial_centroids
	number_iters = 0
	# Train kn
	for _ in range(max_iters):
		fix_u_solve_r(indicator, X, newCentroid)
		fix_r_solve_u(indicator, X, newCentroid)
	# test knn
	predicted_labels = []
	for i in range(N):
		for j in range(K):
			if indicator[j][i] == 1:
				predicted_labels.append(j+1)
				break
	print predicted_labels 
	# return newCentroid, evaluationMatrix
	return newCentroid


def parse_data(file_name):
	f = open(file_name)
	X = []
	labels = []
	for line in f:	
		entry = line.rstrip().split()
		entry = map(lambda x: float(x), entry)
		labels.append(int(entry[-1]))
		X.append(entry[:-1])
	return X, labels


if __name__ == "__main__":
	X, labels = parse_data(sys.argv[1])
	initial_centroids = X[0:3]
	final_centroids =  my_kMeans(X, initial_centroids, 10)
	print labels

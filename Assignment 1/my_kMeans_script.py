from sklearn.manifold import TSNE
from sklearn import metrics
import sys
import numpy as np
import matplotlib.pyplot as plt


def loss_function(indicator, X, centroids):
	K = centroids.shape[0]
	N = X.shape[0]
	cost = 0
	for i in range(N):
		for j in range(K):
			cost += indicator[j][i] * (np.linalg.norm(X[i]-centroids[j]) ** 2)
	return cost


def fix_u_solve_r(indicator, X, centroids):
	K = centroids.shape[0]
	N = X.shape[0]
	for i in range(N):
		arg_min = 0
		for j in range(K):
			if np.linalg.norm(X[i]-centroids[j]) < np.linalg.norm(X[i]-centroids[arg_min]):
				arg_min = j
		for j in range(K):
			if j == arg_min:
				indicator[j][i] = 1
			else:
				indicator[j][i] = 0


def fix_r_solve_u(indicator, X, centroids):
	N = indicator.shape[1]
	K = indicator.shape[0]
	for centroid in range(K):
		denominator = indicator[centroid].sum()
		feature_space = X.shape[1]
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
	main_data = X[:,:-1].astype(float)
	ground_truth = X[:,-1]
	K = initial_centroids.shape[0]
	N = main_data.shape[0]
	# TSNE visualization
	# model = TSNE()
	# tsne_data = model.fit_transform(main_data) #Transformed TSNE data, ready for visualization
	# plt.scatter(tsne_data[:,0], tsne_data[:,1])
	# plt.title('TSNE visualization for given data')
	# plt.show()
	# Create a K*N indicator matrix
	indicator = np.zeros((K, N))
	newCentroid = initial_centroids.astype(float)
	number_iters = 0
	# Train kn
	for _ in range(max_iters):
		fix_u_solve_r(indicator, main_data, newCentroid)
		fix_r_solve_u(indicator, main_data, newCentroid)
		print "Loss function:",loss_function(indicator, main_data, newCentroid)
	# test knn
	predicted_labels = []
	for i in range(N):
		for j in range(K):
			if indicator[j][i] == 1:
				predicted_labels.append(j+1)
				break
	predicted_labels  = np.array(predicted_labels)
	# Metric calculation
	ground_mapping = {}
	modified_ground_mapping = []
	counter = 0.0
	for x in ground_truth:
		if x in ground_mapping:
			modified_ground_mapping.append(ground_mapping[x])
		else:
			counter += 1.0
			ground_mapping[x] = counter
			modified_ground_mapping.append(counter)
	modified_ground_mapping = np.array(modified_ground_mapping)
	MI = metrics.normalized_mutual_info_score(predicted_labels, modified_ground_mapping)
	AMI = metrics.adjusted_mutual_info_score(predicted_labels, modified_ground_mapping)
	RI = metrics.adjusted_rand_score(predicted_labels, modified_ground_mapping)
	ARI = metrics.adjusted_rand_score(predicted_labels, modified_ground_mapping)
	evaluationMatrix = [MI, AMI, RI, ARI]
	return newCentroid, evaluationMatrix


def parse_data(file_name):
	f = open(file_name)
	labels = []
	X = []
	for line in f:
		if line == '\n':
			break
		entry = line.rstrip().replace(',',' ').split()
		X.append(entry)
	return np.array(X)


if __name__ == "__main__":
	X = parse_data(sys.argv[1])
	K = int(sys.argv[2])
	np.random.shuffle(X) #Randomness :)
	initial_centroids = X[:,:-1][:K] # We don't need last column (labels)
	final_centroids, ev_mat =  my_kMeans(X, initial_centroids, 10)
	print ev_mat

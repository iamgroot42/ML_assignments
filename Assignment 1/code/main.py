from sklearn.manifold import TSNE
import sys
import numpy as np
import matplotlib.pyplot as plt
import my_kmeans


# Parse data from file 
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


# Visualize given data using tsne
def visualize_tsne(title, data, labels):
	mapping = {}
	modified_labels = []
	counter = 0.0
	for x in labels:
		if x in mapping:
			modified_labels.append(mapping[x])
		else:
			counter += 1.0
			mapping[x] = counter
			modified_labels.append(counter)
	modified_labels = np.array(modified_labels)
	model = TSNE(learning_rate = 200.0)
	tsne_data = model.fit_transform(data) #Transformed TSNE data, ready for visualization
	plt.scatter(tsne_data[:,0], tsne_data[:,1], c = modified_labels)
	plt.title(title)
	plt.show()


# Predict label for given points, given centroids
def labels_from_centroids(data, centroids):
	K = centroids.shape[0]
	N = data.shape[0]
	predicted_labels = []
	for i in range(N):
		arg_min = 0
		for j in range(K):
			if np.linalg.norm(data[i]-centroids[j]) < np.linalg.norm(data[i]-centroids[arg_min]):
				arg_min = j
		for j in range(K):
			if j == arg_min:
				predicted_labels.append(j+1)
	# Predict labels according to trained K-means
	predicted_labels  = np.array(predicted_labels)
	return predicted_labels


# main()
if __name__ == "__main__":
	if(len(sys.argv) < 3):
		print 'python ' + sys.argv[0] + ' <PATH_TO_FILE>  K'
	X = parse_data(sys.argv[1])
	K = int(sys.argv[2])
	N = X.shape[0] - 1
	np.random.shuffle(X) #Randomness
	initial_centroids = X[:,:-1][:K] # We don't need last column (labels)
	final_centroids, ev_mat =  my_kmeans.my_kMeans(X, initial_centroids, 10)
	# TSNE visualization (before)
	data = X[:,:-1].astype(float)
	print "Original data distribution <graph generated>"
	visualize_tsne(sys.argv[1] + ' : original', data,X[:,-1])
	# TSNE visualization (after)
	print "Data distribution k-means clustering <graph generated>"
	visualize_tsne(sys.argv[1] + ' : with k = ' + str(K), data,labels_from_centroids(data, final_centroids))
	# Average readings over 10 runs
	metrics = [0.0,0.0,0.0,0.0]
	for i in range(10):
		np.random.shuffle(X)
		initial_centroids = X[:,:-1][:K]
		print str(i+1)+"th run"
		final_centroids, ev_mat =  my_kmeans.my_kMeans(X, initial_centroids, 10)
		print ""
		for j in range(4):
			metrics[j] += ev_mat[j]
	for j in range(4):
		metrics[j] /= 10.0
	print "MI:",metrics[0]
	print "AMI:",metrics[1]
	print "RI:",metrics[2]
	print "ARI:",metrics[3]

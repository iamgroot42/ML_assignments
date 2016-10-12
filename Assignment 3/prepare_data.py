import numpy as np
import struct

# Note : code snippet to skip first header for given ubyte file (which contains metadata) taken from
# https://gist.github.com/akesling/5358964


def process_data(trainPath, testPath):
	with open(testPath, 'rb') as labels:
		magic, num = struct.unpack(">II", labels.read(8))
		Y = np.fromfile(labels, dtype = np.int8)
	with open(trainPath, 'rb') as data:
		magic, num, rows, cols = struct.unpack(">IIII", data.read(16))
		X = np.fromfile(data, dtype=np.uint8).reshape(len(Y), rows * cols)
	return X, Y


def sample_data(X, Y, samples_per_class):
	label_wise = {}
	for i in range(len(Y)):
		if Y[i] not in label_wise:
			label_wise[Y[i]] = []
		label_wise[Y[i]].append(X[i])
	X_new = []
	Y_new = []
	for x in label_wise.keys():
		for i in range(samples_per_class):
			X_new.append(label_wise[x][i])
			Y_new.append(x)
	X_new = np.array(X_new)
	Y_new = np.array(Y_new)
	return X_new, Y_new


def data_for_binary_classification(X, Y, classA, classB):
	X_new = []
	Y_new = []
	for i in range(len(Y)):
		if Y[i] == classA:
			X_new.append(X[i])
			Y_new.append(1)
		elif Y[i] == classB:
			X_new.append(X[i])
			Y_new.append(-1)
	X_new = np.array(X_new)
	Y_new = np.array(Y_new)
	return X_new, Y_new


if __name__ == "__main__":
	# get_train_and_test(60000)
	X_train, Y_train = process_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
	X_test, Y_test = process_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
	train_sampled_X, train_sampled_Y = sample_data(X_train, Y_train, 2000)
	test_sampled_X, test_sampled_Y = sample_data(X_test, Y_test, 500)
	data_for_binary_classification(train_sampled_X, train_sampled_Y, 3, 8)[0]

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.externals import joblib
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
	X, Y = shuffle(X, Y)
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
			Y_new.append(classA)
		elif Y[i] == classB:
			X_new.append(X[i])
			Y_new.append(classB)
	X_new = np.array(X_new)
	Y_new = np.array(Y_new)
	return X_new, Y_new


def plot_roc_curve(Y_test, Y_predicted):
	fpr = dict()
	tpr = dict()
	if Y_test.shape[1] == 1:
		fpr[0], tpr[0], _ = roc_curve(Y_test[:,0], Y_predicted[:])
	else:
		for i in range(Y_test.shape[1]):
			fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_predicted[:, i])
	plt.figure()
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	for label in tpr.keys():
		plt.plot(fpr[label], tpr[label], lw = 2, label = 'ROC curve for ' + str(label))
	plt.show()


def save_onevsall(base, model):
	for i in range(len(model.classes_)):
		joblib.dump(model.estimators_[i], base + str(model.classes_[i]) + ".model")

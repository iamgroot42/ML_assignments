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


def plot_roc_curve(fpr, tpr):
	plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
	plt.plot(fpr[0,:],tpr[0,:], 'b.-')
	plt.ylim([0.0, 1.05])
	plt.title('Modified ROC curve')
	plt.xlabel('False Positive Rate')
	plt.ylabel('Classification Accuracy')
	plt.show()


def plot_roc_curve_together(fpr1, tpr1, fpr2, tpr2):
	plt.xlim(min(min(fpr1[0,:]),min(fpr2[0,:])), max(max(fpr1[0,:]),max(fpr2[0,:])))
	plt.ylim([0.0, 1.05])
	plt.plot(fpr1[0,:],tpr1[0,:], 'b.-', color='darkorange', label = 'ROC curve for Linear')
	plt.plot(fpr2[0,:],tpr2[0,:], 'b.-', label = 'ROC curve for RBF')
	plt.title('Modified ROC curve')
	plt.xlabel('False Positive Rate')
	plt.ylabel('Classification Accuracy')
	plt.legend(loc="lower right")
	plt.show()


def save_onevsall(base, model):
	for i in range(len(model.classes_)):
		joblib.dump(model.estimators_[i], base + str(model.classes_[i]) + ".model")
                             

def load_and_test_model(file_path, X_test, Y_test, genCurve = False):
	model = joblib.load(file_path)
	accuracy = model.score(X_test, Y_test)
	if genCurve:
		nROCpts = 100
		tpr = np.zeros([1,nROCpts])
		fpr = np.zeros([1,nROCpts])

		scoreMatrix =  model.decision_function(X_test)
		nTrueLabels = np.count_nonzero(Y_test) 
		nFalseLabels = np.size(Y_test) - nTrueLabels

		minScore = np.min(scoreMatrix)
		maxScore = np.max(scoreMatrix)
		rangeScore = maxScore - minScore
		thdArr = minScore + rangeScore * np.arange(0,1,1.0/(nROCpts))

		for thd_i in range(0,nROCpts):
			thd = thdArr[thd_i]
			ind = np.where(scoreMatrix>=thd) 
			thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
			thisLabel[ind] = 1
			tpr_mat = np.multiply(thisLabel,Y_test)
			tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
			fpr_mat = np.multiply(thisLabel, 1-Y_test)
			fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels
	return accuracy, fpr, tpr

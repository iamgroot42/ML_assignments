import preprocess
from sklearn.svm import LinearSVC
import numpy as np


def SVM_fit_and_predict(X, Y, predict):
	LK = LinearSVC(C = 1.0, max_iter = 1000)
	LK.fit(X, Y)
	return LK.predict(predict)


def accuracy(X_train, Y_train, X_test, Y_test):
	correct_count = 0
	Y_predicted = SVM_fit_and_predict(X_train, Y_train, X_test)
	for i in range(len(Y_predicted)):
		if Y_predicted[i] == Y_test[i]:
			correct_count += 1
	return (100.0 * (correct_count/float(len(Y_predicted))))


def k_fold_cross_validation(X, Y, grid_params):
	k = len(grid_params)
	start = 0
	eff_k = int((k * len(Y))/100.0)
	accuracies = []
	for i in range(k):
		# Split data
		left_data = X[:start]
		right_data = X[start + eff_k:]
		modified_data = np.append(left_data, right_data, axis=0)
		# Split labels
		left_labels = Y[:start]
		right_labels = Y[start + eff_k:]
		modified_labels = left_labels
		modified_labels = np.append(left_labels, right_labels, axis=0)
		# Validation data
		validation_data = X[start:start + eff_k]
		validation_labels = Y[start:start + eff_k]
		accuracies.append(accuracy(modified_data, modified_labels, validation_data, validation_labels))
		start += eff_k
	return accuracies


if __name__ == "__main__":
	X_train, Y_train = preprocess.process_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
	X_test, Y_test = preprocess.process_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
	train_sampled_X, train_sampled_Y = preprocess.sample_data(X_train, Y_train, 2000)
	test_sampled_X, test_sampled_Y = preprocess.sample_data(X_test, Y_test, 500)
	EX, EY = preprocess.data_for_binary_classification(train_sampled_X, train_sampled_Y, 3, 8)
	EX_, EY_ = preprocess.data_for_binary_classification(test_sampled_X, test_sampled_Y, 3, 8)
	print accuracy(EX, EY, EX_, EY_)
	print k_fold_cross_validation(EX, EY, [1.0,1.5,2.0,2.5,3.0])

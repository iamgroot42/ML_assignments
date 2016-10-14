import misc
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Ignore warnings based on convergence because of number of iterations
warnings.filterwarnings("ignore")


def SVM_fit_and_predict(X_train, Y_train, X_test, Y_test, c, kern, gam, plotROC = False):
	# -1 : use all CPUs
	print "Trying with",c
	digits = np.unique(Y_train)
	multi_Y_train = label_binarize(Y_train, classes = digits)
	multi_Y_test = label_binarize(Y_test, classes = digits)
	SVM_K = OneVsRestClassifier(SVC(C = c, max_iter = 100, kernel = kern, gamma = gam), -1)
	SVM_K.fit(X_train, multi_Y_train)
	if plotROC:
		misc.plot_roc_curve(multi_Y_test, SVM_K.predict(X_test))
	return SVM_K.score(X_test, multi_Y_test)


def k_fold_cross_validation(X, Y, k, model, C, kernel, gamma):
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
		accuracies.append(model(modified_data, modified_labels,validation_data, 
			validation_labels, C, kernel, gamma))
		start += eff_k
	print np.mean(accuracies)
	return np.mean(accuracies)


def grid_search(X, Y, k, model, kernel, grid1, grid2 = ['auto']):
	opt_val = (grid1[0], grid2[0])
	opt_acc = 0.0
	for gamma in grid2:
		accuracies = []
		for C in grid1:
			accuracy = k_fold_cross_validation(X, Y, k, model, C, kernel, 'auto')
			accuracies.append(accuracy)
			if accuracy > opt_acc:
				opt_acc = accuracy
				opt_val = (C, gamma)
		plt.figure()
		plt.xlabel('Value of C')
		plt.ylabel(str(k) + '-fold Cross validation accuracy')
		plt.title('Accuracy v/s C, for gamma = ' + str(gamma))
		plt.legend(loc="lower right")
		plt.plot(grid1, accuracies, color='darkorange', lw = 2)
		plt.show()
	return opt_val


if __name__ == "__main__":
	X_train, Y_train = misc.process_data("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte")
	X_test, Y_test = misc.process_data("MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte")
	train_sampled_X, train_sampled_Y = misc.sample_data(X_train, Y_train, 2000)
	test_sampled_X, test_sampled_Y = misc.sample_data(X_test, Y_test, 500)
	EX, EY = misc.data_for_binary_classification(train_sampled_X, train_sampled_Y, 3, 8)
	EX_, EY_ = misc.data_for_binary_classification(test_sampled_X, test_sampled_Y, 3, 8)
	# print k_fold_cross_validation(EX, EY, 5, SVM_fit_and_predict, 1.0, 'linear', 'auto')
	# gamma = 0.01
	# print k_fold_cross_validation(EX, EY, 5, SVM_fit_and_predict, 1.0, 'rbf', gamma)
	opt_C, opt_gamma = grid_search(EX, EY, 5, SVM_fit_and_predict, 'linear', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
	SVM_fit_and_predict(EX, EY, EX_, EY_, opt_C, 'linear', opt_gamma, True)
	# print grid_search(EX, EY, 5, SVM_fit_and_predict, 'rbf', [1e-3, 1, 1e3], [1e-3, 1, 1e3])

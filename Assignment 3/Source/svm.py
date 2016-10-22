import misc
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import warnings

# Ignore warnings based on convergence because of number of iterations
warnings.filterwarnings("ignore")


def SVM_fit_and_predict(X_train, Y_train, X_test, Y_test, c, kern, gam, plotROC = False):
	# -1 : use all CPUs
	digits = np.unique(Y_train)
	multi_Y_train = label_binarize(Y_train, classes = digits)
	multi_Y_test = label_binarize(Y_test, classes = digits)
	SVM_K = OneVsRestClassifier(SVC(C = c, max_iter = 1000, kernel = kern, gamma = gam, verbose = True), -1)
	SVM_K.fit(X_train, multi_Y_train)
	if plotROC:
		misc.plot_roc_curve(multi_Y_test, SVM_K.predict(X_test))
	return SVM_K.score(X_test, multi_Y_test), SVM_K


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
			validation_labels, C, kernel, gamma)[0])
		start += eff_k
	mean_accuracy = np.mean(accuracies)
	print "Cross validation accuracy for (",str(C),",",str(gamma),") :",mean_accuracy
	return mean_accuracy


def grid_search(X, Y, k, model, kernel, grid1, grid2 = ['auto'], plot = False):
	opt_val = (grid1[0], grid2[0])
	opt_acc = 0.0
	for gamma in grid2:
		accuracies = []
		for C in grid1:
			accuracy = k_fold_cross_validation(X, Y, k, model, C, kernel, gamma)
			accuracies.append(accuracy)
			if accuracy > opt_acc:
				opt_acc = accuracy
				opt_val = (C, gamma)
		if plot:
			plt.figure()
			plt.xlabel('Value of C')
			plt.ylabel(str(k) + '-fold Cross validation accuracy')
			plt.title('Accuracy v/s C, for gamma = ' + str(gamma))
			plt.legend(loc="lower right")
			plt.plot(grid1, accuracies, color = 'darkorange', lw = 2)
			plt.show()
	return opt_val


def training_phase(X_train, Y_train, X_test, Y_test):
	C_grid = [1e-7, 1e-3, 1e1, 1e5]
	gamma_grid = [1e-9, 1e-6, 1e-3]
	# Part(a) : 3/8 binary classification
	train_sampled_X, train_sampled_Y = misc.sample_data(X_train, Y_train, 2000)
	test_sampled_X, test_sampled_Y = misc.sample_data(X_test, Y_test, 500)
	EX, EY = misc.data_for_binary_classification(train_sampled_X, train_sampled_Y, 3, 8)
	EX_, EY_ = misc.data_for_binary_classification(test_sampled_X, test_sampled_Y, 3, 8)
	opt_C, opt_gamma = grid_search(EX, EY, 5, SVM_fit_and_predict, 'linear', C_grid)
	test_accuracy, SVM_OBJ = SVM_fit_and_predict(EX, EY, EX_, EY_, opt_C, 'linear', opt_gamma, True)
	print "Test accuracy for (",str(opt_C),",",str(opt_gamma),") :",test_accuracy
	joblib.dump(SVM_OBJ, "../Models/model_linear.model")
	# Part(b) : multi-class classification
	opt_C, opt_gamma = grid_search(train_sampled_X, train_sampled_Y, 5, SVM_fit_and_predict, 'linear', C_grid)
	test_accuracy, SVM_OBJ_2 = SVM_fit_and_predict(train_sampled_X, train_sampled_Y, test_sampled_X, test_sampled_Y, opt_C, 'linear', opt_gamma, True)
	print "Test accuracy for (",str(opt_C),",",str(opt_gamma),") :",test_accuracy
	joblib.dump(SVM_OBJ_2, "../Models/multi.model")
	# misc.save_onevsall("../Models/multi", SVM_OBJ_2)
	# Part(c) : RBF multi-class classification
	opt_C, opt_gamma = grid_search(train_sampled_X, train_sampled_Y, 5, SVM_fit_and_predict, 'rbf', C_grid, gamma_grid)
	test_accuracy, SVM_OBJ_3 = SVM_fit_and_predict(train_sampled_X, train_sampled_Y, test_sampled_X, test_sampled_Y, opt_C, 'rbf', opt_gamma, True)
	print "Test accuracy for (",str(opt_C),",",str(opt_gamma),") :",test_accuracy
	misc.save_onevsall("../Models/rbf", SVM_OBJ_3)
	# joblib.dump(SVM_OBJ_3, "../Models/rbf.model")


def testing_phase(X_test, Y_test):
	binary_digits = [3,8]
	digits = [0,1,2,3,4,5,6,7,8,9]
	EX_, EY_ = misc.data_for_binary_classification(X_test, Y_test, 3, 8)
	binarized_EY_ = label_binarize(EY_, classes = binary_digits)
	binarized_Y_test = label_binarize(Y_test, classes = digits)
	# Part(a) : 3/8 binary classification
	acc1,fpr1,tpr1 = misc.load_and_test_model("../Models/model_linear.model", EX_, binarized_EY_, True)
	print "Test accuracy for [3,8] linear:", str(acc1)
	misc.plot_roc_curve(fpr1,tpr1)
	# Part(b) : multi-class classification
	acc2,fpr2,tpr2 = misc.load_and_test_model("../Models/multi.model", X_test, binarized_Y_test, True)
	print "Test accuracy for multi-linear:", str(acc2)
	misc.plot_roc_curve(fpr2,tpr2)
	# Part(c) : RBF multi-class classification
	acc3,fpr3,tpr3 = misc.load_and_test_model("../Models/rbf.model", X_test, binarized_Y_test, True)
	print "Test accuracy for multi-rbf:", str(acc3)
	misc.plot_roc_curve(fpr3,tpr3)
	misc.plot_roc_curve_together(fpr2,tpr2,fpr3,tpr3)


if __name__ == "__main__":
	# Process data
	X_train, Y_train = misc.process_data("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte")
	X_test, Y_test = misc.process_data("MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte")
	test_sampled_X, test_sampled_Y = misc.sample_data(X_test, Y_test, 500)
	# Train models 
	# training_phase(X_train, Y_train, X_test, Y_test)
	# Test models
	testing_phase(X_test, Y_test)
	
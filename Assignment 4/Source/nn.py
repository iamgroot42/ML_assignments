from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Reference for auto-encoder : https://blog.keras.io/building-autoencoders-in-keras.html

def anti_onehot(x):
	return np.argmax(x, axis=1)


def plot_conf_mat(x):
	x = x / x.sum(axis=1).astype(float)
	plt.matshow(x)
	plt.colorbar()
	plt.show()


def feed_forward_nn(ne, bs, filepath=None):
	# Load MNIST data
	(X_train, y_train), (X_test, y_test) =  mnist.load_data()
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train =  X_train.reshape(len(X_train), X_train.shape[1] * X_train.shape[2])
	X_test =  X_test.reshape(len(X_test), X_test.shape[1] * X_test.shape[2])
	feature_size = X_train.shape[1]
	# Create model
	model = Sequential()
	model.add(Dense(output_dim=500, input_dim=feature_size))
	model.add(Activation("tanh"))
	model.add(Dense(output_dim=250))
	model.add(Activation("tanh"))
	model.add(Dense(output_dim=10))
	model.add(Activation("softmax"))
	# Configure model
	model.compile(loss='categorical_crossentropy',
		optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-8, decay=0.0),
		metrics=['accuracy'])
	# Fit model
	model.fit(X_train,y_train,
				nb_epoch=ne,
				batch_size=bs,
				callbacks=[TensorBoard(log_dir='/tmp/ffnn')])
	if filepath:
		model.save(filepath)
	# Evaluate performance
	loss_and_metrics = model.evaluate(X_test, y_test, batch_size=bs)
	print "\n"
	print "Classification accuracy:", loss_and_metrics[1]
	return confusion_matrix(anti_onehot(y_test), \
			anti_onehot(np_utils.to_categorical(model.predict_classes(X_test, batch_size=bs))), \
			np.unique(anti_onehot(y_test)))


def auto_encoder(learning_rate, aune, ne, bs, autoencoder_weight_file):
	# Load MNIST data
	(X_train, y_train), (X_test, y_test) =  mnist.load_data()
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train =  X_train.reshape(len(X_train), X_train.shape[1] * X_train.shape[2]).astype(float)
	X_test =  X_test.reshape(len(X_test), X_test.shape[1] * X_test.shape[2]).astype(float)
	feature_size = X_train.shape[1]
	# Normalize data
	X_train /= X_train.max()
	X_test /= X_test.max()
	# Create autoencoder
	input_img = Input(shape=(feature_size,))
	encoded = Dense(100, activation='sigmoid')(input_img)
	decoded = Dense(feature_size, activation='sigmoid')(encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Create encoder
	encoder = Model(input=input_img, output=encoded)
	# Configure autoencoder
	autoencoder.compile(loss='binary_crossentropy',optimizer='Adadelta')
	autoencoder.fit(X_train, X_train,
				nb_epoch=aune,
				batch_size=bs,
				validation_data=(X_test, X_test))
	return autoencoder
	# Compress input data according to learnt auto-encoder
	X_train_comp = encoder.predict(X_train, batch_size=bs)
	X_test_comp = encoder.predict(X_test, batch_size=bs)
	# Create ffnn
	model = Sequential()
	model.add(Dense(output_dim=50, input_dim=100))
	model.add(Activation("sigmoid"))
	model.add(Dense(output_dim=10))
	model.add(Activation("softmax"))
	# Configure model
	model.compile(loss='categorical_crossentropy',
		optimizer=Adadelta(lr=learning_rate),
		metrics=['accuracy'])
	# Fit model
	model.fit(X_train_comp,y_train,
				nb_epoch=ne,
				batch_size=bs,
				callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
	# Evaluate performance
	loss_and_metrics = model.evaluate(X_test_comp, y_test, batch_size=bs)
	print "\n"
	print "Classification accuracy:", loss_and_metrics[1]


def vis_weights(model):
	model_number = 1
	for x in model.get_weights():
		z = x
		if len(x.shape) == 1:
			z = np.reshape(x, (x.shape[0], 1))
		plt.matshow(z, fignum=100, cmap=plt.cm.gray)
		savefig(str(model_number) + ".png")
		model_number += 1


if __name__ == "__main__":
	# Question 3
	ffnn_cm = feed_forward_nn(10, 128, "FFNN_Adadelta_0.1")
	plot_conf_mat(ffnn_cm)
	# Question 4
	m = auto_encoder(7, 20, 30, 128, 'autoencoder_weights.png')
	vis_weights(m)

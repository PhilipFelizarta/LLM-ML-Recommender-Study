import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

# Two objectives, design MLP and 1D CNN neural networks. 
def NN_classifier(input_num, output_num, lr=3e-3, gamma=1e-5, verbose=False, hidden_neurons=64, hidden_layers=1):
	'''
	input_num: Number of input movies.
	We assume in this project there will 3 features per movie.

	output_num: Number of output movies. The model will recommend one using softmax.

	lr = learning rate
	gamma = l2 normalization constant

	hidden_neurons = number of neurons per hidden layer
	hidden_layers = number of hidden layers.

	verbose = True or False, print model summary
	'''

	X = Input(shape=(input_num, 3)) # Movie matrix input

	# Standard neural network will just flatten
	hidden = Flatten()(X)

	for h in range(hidden_layers): # Construct hidden layer graph
		hidden = Dense(hidden_neurons, activation="relu", kernel_regularizer=l2(gamma))(hidden)

	# Output Layer, classification task
	classifier = Dense(output_num, activation="softmax", kernel_regularizer=l2(gamma))(hidden)

	# Create model object
	model = Model(inputs=X, outputs=classifier)

	# Tune optimizer
	opt = SGD(learning_rate=lr)

	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	if verbose: # Don't want this everytime, since we will tune using gaussian process.
		model.summary()

	return model

def CNN_classifier(input_num, output_num, lr=3e-3, gamma=1e-5, verbose=False, 
				   hidden_neurons=64, hidden_layers=1, filters=32, kernel_size=3, pool_size=2):
	'''
	input_num: Number of input movies.
	We assume in this project there will 3 features per movie.

	output_num: Number of output movies. The model will recommend one using softmax.

	lr = learning rate
	gamma = l2 normalization constant

	hidden_neurons = number of neurons per hidden layer
	hidden_layers = number of hidden layers.

	filters = number of filters for the convolutional layers
	kernel_size = size of the kernel for the convolutional layers
	pool_size = size of the max pooling window

	verbose = True or False, print model summary
	'''

	X = Input(shape=(input_num, 3)) # Movie matrix input

	# Convolutional layer
	hidden = Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", 
					kernel_regularizer=l2(gamma))(X)
	hidden = MaxPooling1D(pool_size=pool_size)(hidden)

	# Flatten the output from convolutional layers
	hidden = Flatten()(hidden)

	for h in range(hidden_layers): # Construct hidden layer graph
		hidden = Dense(hidden_neurons, activation="relu", kernel_regularizer=l2(gamma))(hidden)

	# Output Layer, classification task
	classifier = Dense(output_num, activation="softmax", kernel_regularizer=l2(gamma))(hidden)

	# Create model object
	model = Model(inputs=X, outputs=classifier)

	# Tune optimizer
	opt = SGD(learning_rate=lr)

	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	if verbose: # Don't want this everytime, since we will tune using gaussian process.
		model.summary()

	return model

def XGB_classifier(n_estimators=200, max_depth=3, learning_rate=0.1, gamma=0.0):
	model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
		learning_rate=learning_rate, gamma=gamma, use_label_encoder=False, eval_metric='mlogloss')
	return model
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
from keras import backend as keras_backend
from keras import layers
from metalearning_utils import MSE_loss, copy_model, compute_MSE_loss
import metalearning
import trainer
import models
import data_processing

# Load in MATLAB-generated data
X_train, y_train = data_processing.process_MATLAB_data(
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/PSF_2_0to1in9_100.mat',
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/Zpos_2_0to1in9_100.mat',
    normalise_images=False)

X, y = data_processing.split_MATLAB_data((X_train, y_train * 10), mags=np.linspace(0, 1, 9),
                                         zpos=np.linspace(-1, 1, 11), iterations=100)

# Process data, need to collapse this into an utility function
keys = list(X.keys())
random.shuffle(keys)
training_keys, test_key = keys[0:-1], [keys[-1]]
print(training_keys, test_key)

X_train = {key: X[key] for key in training_keys}
X_train = list(X_train.values())
X_train = np.concatenate(X_train)

y_train = {key: y[key] for key in training_keys}
y_train = list(y_train.values())
y_train = np.concatenate(y_train)

X_test = {key: X[key] for key in test_key}
X_test = list(X_test.values())
X_test = np.concatenate(X_test)

y_test = {key: y[key] for key in test_key}
y_test = list(y_test.values())
y_test = np.concatenate(y_test)

lr_list = [0.1, 0.01, 0.001, 1e-3, 1e-4, 1e-5]
train_losses = np.zeros((1,1000))
val_losses = np.zeros((1,1000))

for inner_rate in lr_list:
    for meta_rate in lr_list:
        tf.keras.backend.set_floatx('float64')
        rep_model = models.DenseModel()
        rep_model(X_train[0:5])  # Initialise weights

        history = metalearning.train_REPTILE_simple(rep_model, (X, y), training_keys=training_keys,
                                                    epochs=5, lr_inner=inner_rate,
                                                    batch_size=32, lr_meta=meta_rate)

        train_loss = np.array(history['loss'])
        val_loss = np.array(history['val_loss'])

        train_losses = np.append(train_losses, [train_loss], axis=0)
        val_losses = np.append(val_losses, [val_loss], axis=0)

np.save('train_losses', train_losses)
np.save('val_losses', val_losses)


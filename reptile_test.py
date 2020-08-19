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
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/PSF_2_0to1in9_2in201_100.mat',
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/Zpos_2_0to1in9_2in201_100.mat',
    normalise_images=False)

X, y = data_processing.split_MATLAB_data((X_train, y_train * 10), mags=np.linspace(0, 1, 9),
                                         zpos=np.linspace(-2, 2, 201), iterations=100)

keys = list(X.keys())
random.shuffle(keys)
training_keys = keys


rep_model = models.create_DenseModel()

start = time.time()
history = metalearning.train_REPTILE_simple(rep_model, (X, y), training_keys=training_keys,
                                            epochs=1000, lr_inner=1e-3,
                                            batch_size=32, lr_meta=1e-4,
                                            logging=1, show_plot=False)
end = time.time()

f = open('logs.txt', 'w')
f.write("Time taken: ")
f.write(str(end-start))
f.close()

train_loss = np.array(history['loss'])
val_loss = np.array(history['val_loss'])

np.save('train_loss', train_loss)
np.save('val_loss', val_loss)

rep_model.save('rep_model')

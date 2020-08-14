import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
import trainer
import models
import data_processing
from keras import backend as keras_backend
from metalearning_utils import MSE_loss, copy_model, compute_MSE_loss
import test_tools as tt

f = open("setup.txt", "w")
f.write("Using acceptable data from blob_processed zstack, only 46 in total \n"
        "Test-Train split of 50-50 for a restrictive environment \n"
        "Comparing trained_model_dense and trained_model_reptile \n"
        "Testing epochs = np.append([1, 10], np.arange(50,1001, 50))")

acceptable = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
              147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
              168, 169, 170, 171]

X = np.load('/rds/general/user/rsa14/home/blobs/psf.npy')
y = np.load('/rds/general/user/rsa14/home/blobs/pos.npy')

X_ = X[acceptable]
y_ = y[acceptable]

test_epochs = np.append([1, 10], np.arange(50, 1001, 50))

dense_model = keras.models.load_model('/rds/general/user/rsa14/home/cnnSTORM/saved_models/trained_model_Dense')
dense_model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

rep_model = keras.models.load_model('/rds/general/user/rsa14/home/cnnSTORM/saved_models/trained_model_reptile_1000')


results = tt.compare_models_finetuned([dense_model, rep_model], (X_, y_/100), epochs=test_epochs,
                                      repetitions=100, test_train_split=0.5, batch_size=4)

np.save('results', results)


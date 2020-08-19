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

dataset = '2_2in51'

f = open("setup.txt", "w")
f.write("Dataset: " + str(dataset))
f.write("Using all data from blob_processed zstack\n"
        "Test-Train split of 50-50 for a restrictive environment \n"
        "Comparing oracle, control, rep and dense \n"
        "Testing epochs = np.append([1, 10], np.arange(50,1001, 50)), 10 iterations")
f.close()

test_epochs = np.append([1, 10], np.arange(50, 1001, 50))
X = np.load('/rds/general/user/rsa14/home/blobs/psf.npy')
y = np.load('/rds/general/user/rsa14/home/blobs/pos.npy')

X_train, y_train, X_test, y_test = data_processing.split_test_train((X, y / 100), split=0.5)
control_model = models.create_DenseModel()
history_control = trainer.train_model(control_model, x_train=X_train, y_train=y_train,
                                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                      loss='mse', metrics=None, validation_split=0.2,
                                      epochs=1000, batch_size=32, summary=False, verbose=0)

oracle_model = keras.models.load_model('/rds/general/user/rsa14/home/results/oracle_model/model')

dense_model = keras.models.load_model('/rds/general/user/rsa14/home/results/' + dataset +
                                      '/dense_model/model')
rep_model = keras.models.load_model('/rds/general/user/rsa14/home/results/' + dataset +
                                    '/rep_model/model')

oracle_model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
rep_model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
dense_model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# Pre-trained results:
oracle_mse = oracle_model.evaluate(X_test, y_test, batch_size=32)
control_mse = control_model.evaluate(X_test, y_test, batch_size=32)
rep_mse = rep_model.evaluate(X_test, y_test, batch_size=32)
dense_mse = dense_model.evaluate(X_test, y_test, batch_size=32)

pretrained_results = np.array([oracle_mse, control_mse, rep_mse, dense_mse])

models = [control_model, rep_model, dense_model]

results = tt.compare_models_finetuned(models, (X_train, y_train), (X_test, y_test), epochs=test_epochs,
                                      repetitions=10, batch_size=32)

np.save('finetune_comparison', results)
np.save('pretrained', pretrained_results)

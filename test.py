import numpy as np
import matplotlib.pyplot as plt
import models
import resnet_models
import trainer
import data_processing
import keras
import tensorflow as tf

# print(tf.config.list_physical_devices())
# Run process_data first to get cached image data + z-positions
# STORM data
# X_train, y_train = data_processing.process_STORM_data('ThunderSTORM/Simulated/LowDensity',
#                                                       samples=500)
# MATLAB data
X_train, y_train = data_processing.process_MATLAB_data(
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/PSF_2_all.mat',
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/Zpos_2_all.mat',
    normalise_images=False)
# print(X_train.shape)


# Rescale z-pos
# y_train = y_train*10
# y_train_norm, y_mean, y_sd = data_processing.scale_zpos(y_train)
# print(X_train.shape)



resmodel = resnet_models.create_resnet()

history = trainer.train_model(resmodel, x_train=X_train, y_train=y_train,
                              optimizer=keras.optimizers.Adam(learning_rate=0.001),
                              loss='mse', metrics=['mse'], validation_split=0.2,
                              epochs=25, batch_size=32, summary=False)

records = trainer.get_metrics(history, show_plot=True, plot_name='resmodel.png')
# print(records)

f = open('dict.txt', 'w')
f.write(str(history.history))
f.close()



import numpy as np
import matplotlib.pyplot as plt
import models
import resnet_models
import trainer
import data_processing
import keras
import smNet
import tensorflow as tf
import time

# print(tf.config.list_physical_devices())
# Run process_data first to get cached image data + z-positions
# STORM data
# X_train, y_train = data_processing.process_STORM_data('ThunderSTORM/Simulated/LowDensity',
#                                                       samples=500)
# # MATLAB data
X_train, y_train = data_processing.process_MATLAB_data(
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/PSF_2to6_0to1in9_2in51_100.mat',
    '/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/Zpos_2to6_0to1in9_2in51_100.mat',
    normalise_images=False)

densemodel = models.create_DenseModel()
densemodel.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mse')
checkpoint_filepath = 'model'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


def scheduler(epoch, lr):
    if epoch <= 300:
        return lr

    elif epoch > 300 and epoch <= 700:
        return 1e-4

    else:
        return 1e-5

lr = tf.keras.callbacks.LearningRateScheduler(scheduler)


start = time.time()
history_dense = densemodel.fit(X_train, y_train * 10, validation_split=0.2, epochs=1000, batch_size=32, verbose=1,
                               callbacks=[model_checkpoint_callback, lr])
end = time.time()
dense_model_1000_train = np.array(history_dense.history['loss'])
dense_model_1000_val = np.array(history_dense.history['val_loss'])

np.save('train_loss', dense_model_1000_train)
np.save('val_loss', dense_model_1000_val)

f = open("setup.txt", "w")
f.write("2to6_2in51 data")
f.write(str(end - start))
f.close()

# # print(X_train.shape)
#
#
# # Rescale z-pos
# # y_train = y_train*10
# # y_train_norm, y_mean, y_sd = data_processing.scale_zpos(y_train)
# # print(X_train.shape)
#
#
# model = smNet.create_smNet(min_z=-1.0, max_z=1.0)
# resmodel = resnet_models.create_resnet()
#
# history = trainer.train_model(model, x_train=X_train, y_train=y_train,
#                               optimizer=keras.optimizers.Adam(learning_rate=0.001),
#                               loss='mse', metrics=None, validation_split=0.2,
#                               epochs=5, batch_size=32, summary=False)
#
# records = trainer.get_metrics(history, show_plot=True, plot_name='smNet.png')

# d = {'a': [1, 2, 3, 4, 5], 'b': [1, 2]}
# a = np.array([1,2,3,4,5])
# b = np.array([6,7,8,9,10])
# c = np.array([0.2,0.4,1,2,3])
# dt = np.array([a,b])
#
# np.save('a', a)
#  np.save('b', b)
# f = open("setup.txt", "w")
# f.write("yada yada yada, \n"
#         "wtf \n"
#         "and hi \n"
#         "f")
# f.close()

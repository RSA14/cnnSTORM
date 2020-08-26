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

datasetIm = ['PSF_2to6_0to1in9_2in201_100_1.mat', 'PSF_2to6_0to1in9_2in201_100_2.mat',
             'PSF_2to6_0to1in9_2in201_100_3.mat', 'PSF_2to6_0to1in9_2in201_100_4.mat']
datasetZ = 'Zpos_2to6_0to1in9_2in201_100.mat'
model = models.create_DenseModel()
checkpoint_filepath = 'model'
paths = []

for path in datasetIm:
    paths.append(f'/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/{path}')

X_train, y_train = data_processing.process_multiple_MATLAB_data(
    paths,
    f'/rds/general/user/rsa14/home/data/Simulated/PSF_toolbox/astig/test/{datasetZ}',
    normalise_images=False)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mse')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

start = time.time()
history_dense = model.fit(X_train, y_train * 10, validation_split=0.2, epochs=1000, batch_size=32, verbose=1,
                          callbacks=[model_checkpoint_callback])
end = time.time()
dense_model_1000_train = np.array(history_dense.history['loss'])
dense_model_1000_val = np.array(history_dense.history['val_loss'])

np.save('train_loss', dense_model_1000_train)
np.save('val_loss', dense_model_1000_val)

f = open("setup.txt", "w")
f.write(f"{datasetZ} data")
f.write(str(end - start))
f.close()

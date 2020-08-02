import numpy as np
import matplotlib.pyplot as plt
import models
import resnet_models
import trainer
import data_processing
import keras
import tensorflow as tf

print(tf.config.list_physical_devices())
# Run process_data first to get cached image data + z-positions
#STORM data
# X_train, y_train = data_processing.process_STORM_data('ThunderSTORM/Simulated/LowDensity',
#                                                       samples=500)
#MATLAB data
X_train, y_train = data_processing.process_MATLAB_data('../data/Simulated/PSF_toolbox/PSF_2_all.mat',
                                                       '../data/Simulated/PSF_toolbox/Zpos_2_all.mat',
                                                       normalise_images=False)
print(X_train.shape)


# Rescale z-pos
# y_train = y_train*10
# y_train_norm, y_mean, y_sd = data_processing.scale_zpos(y_train)
print(X_train.shape)

#Create resnet model

inputs = keras.layers.Input((32,32,1))
model = resnet_models.conv_bn_activation_block(inputs, 64, activation = 'prelu', kernel_size = 5, strides = (1,1))

model = resnet_models.bottleneck_projection_block(model, filters = [64,64,256], strides=1, activation ='prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
# model = keras.layers.MaxPooling2D(2)(model)

model = resnet_models.bottleneck_projection_block(model, filters = [64,64,256], strides=2, activation='prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
# model = keras.layers.MaxPooling2D(2)(model)

model = resnet_models.bottleneck_projection_block(model, filters = [64,64,256], strides=2, activation='prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')
model = resnet_models.bottleneck_identity_block(model, filters = [64,64,256], activation = 'prelu')


model = keras.layers.GlobalAveragePooling2D()(model)
model = keras.layers.Flatten()(model)
outputs = keras.layers.Dense(1)(model)
resnet_1 = keras.Model(inputs, outputs)
# resnet_1.summary()

history = trainer.train_model(resnet_1, x_train=X_train, y_train=y_train,
                              optimizer=keras.optimizers.Adam(learning_rate=0.001),
                              loss='mse', metrics=['mse'], validation_split=0.2,
                              epochs=50, batch_size=32, summary=False)

records = trainer.get_metrics(history, show_plot=True)
print(records)



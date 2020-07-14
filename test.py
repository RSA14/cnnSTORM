import numpy as np
import matplotlib.pyplot as plt
from models import conv_1, resnet_model, densemodel, conv_test
import trainer
import data_processing
# from generate_data import X_train, y_train
import keras

# Run process_data first to get cached image data + z-positions
#STORM data
# X_train, y_train = data_processing.process_STORM_data('ThunderSTORM/Simulated/LowDensity',
#                                                       samples=500)
#MATLAB data
X_train, y_train = data_processing.process_MATLAB_data('PSF_ast.mat', 'Zpos_ast.mat', normalise_images=True)

# Rescale z-pos
# y_train = y_train*10
# y_train_norm, y_mean, y_sd = data_processing.scale_zpos(y_train)
print(X_train.shape)

history = trainer.train_model(conv_test, x_train=X_train, y_train=y_train,
                              optimizer=keras.optimizers.Adam(learning_rate=0.01),
                              loss='mse', metrics=['mse'], validation_split=0.2,
                              epochs=20, batch_size=32, summary=False)

records = trainer.get_metrics(history, show_plot=True)
print(records)



import numpy as np
import matplotlib.pyplot as plt
from models import conv_1
import trainer
import data_processing
# from generate_data import X_train, y_train
import keras

# Run generate_data first to get cached image data + z-positions
X_train, y_train = data_processing.generate_STORM_data('ThunderSTORM/Simulated/LowDensity',
                                                       samples=500)


# Rescale z-pos
y_train_norm, mean, sd = data_processing.scale_zpos(y_train)
print(X_train.shape)

history = trainer.train_model(conv_1, x_train=X_train, y_train=y_train_norm,
                              optimizer=keras.optimizers.Adam(),
                              loss='mse', metrics=['mse'], validation_split=0.2,
                              epochs=10, batch_size=32)

records = trainer.get_metrics(history, show_plot=True)
print(records)

# print(history.history.keys())
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

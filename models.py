import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Basic CNN
model = Sequential()
model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(4, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1))





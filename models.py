import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Conv2D, Flatten, Dense, Add


# General CNN generator

def create_CNN(conv, batch_norm, activation, pooling, dense, dropout,
               dense_activation=None, input_shape=(32, 32, 1), output_shape=1,
               global_avg_pool=True, filter_size=3,
               show_summary=False):
    """
    Constructs a CNN using repeated blocks of: Conv2D -> BatchNorm -> Activation -> Pooling
    FCN layer block constructed using dense layer -> dense activation -> dropout (if present)
    Create these "blocks", using "None" values to represent an absent layer of that type in a particular block

    :param conv: List of filters for convolution layers, use "None" to signify an unused conv layer
    :param batch_norm: List of 1 or 0 for Batch Normalisation layers present or absent per block
    :param activation:'prelu', 'relu', 'tanh', 'sigmoid' activation layers
    :param pooling: List of tuples of pooling window sizes
    :param dense: List/tuple of neurons/size of dense connected layers
    :param dropout: List containing dropout rates or None after each dense layer. Must be same length as dense.
    :param dense_activation: Common activation function for dense layers
    :param input_shape: Input shape to CNN, default image is (32, 32, 1)
    :param output_shape: Output of CNN, default is 1 for z-pos prediction
    :param global_avg_pool: Boolean indicating whether to use global avg pooling at the end before dense
    :param filter_size: Convolution kernel size, usually fixed to 3
    :param show_summary: Displays model summary
    :return: CNN model object
    """
    if dense is None:
        dense = [128, 64, 32]
    if len(conv) != len(batch_norm) != len(activation) != len(pooling):
        return print("Lists specifying each type of layer must be of the same length.")
    if len(dense) != len(dropout):
        return print("Lists specifying dense layers and dropout layers must be of the same length.")

    inputs = keras.Input(shape=input_shape)
    model = inputs  # Instantiate model

    for i in range(len(conv)):
        if conv[i] is not None:
            model = layers.Conv2D(filters=conv[i], kernel_size=filter_size, kernel_regularizer='l2')(model)

        if batch_norm[i] is not None:
            model = layers.BatchNormalization()(model)

        if activation[i] is not None:
            if activation[i] == 'prelu':
                model = layers.PReLU()(model)

            elif activation[i] == 'relu':
                model = layers.ReLU()(model)

            elif activation[i] == 'tanh':
                model = keras.activations.tanh(model)

            elif activation[i] == 'sigmoid':
                model = keras.activations.sigmoid(model)

        if pooling[i] is not None:
            model = layers.MaxPooling2D(pool_size=pooling[i])(model)

    # Global Avg Pooling at the end:
    if global_avg_pool:
        model = layers.GlobalAveragePooling2D()(model)

    # Flatten output for FCN layers
    model = layers.Flatten()(model)

    for i in range(len(dense)):
        if dense[i] is not None:
            model = layers.Dense(dense[i], activation=dense_activation)(model)
        if dropout[i] is not None:
            model = layers.Dropout(dropout[i])(model)

    outputs = layers.Dense(output_shape)(model)

    cnn_model = keras.Model(inputs=inputs, outputs=outputs)

    if show_summary:
        cnn_model.summary()

    return cnn_model

# Model and utils for metalearning
class DenseModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = layers.Flatten(input_shape=(32, 32, 1), dtype=tf.float64)
        self.hidden1 = layers.Dense(256, dtype=tf.float64)
        self.hidden2 = layers.Dense(128, dtype=tf.float64)
        self.hidden3 = layers.Dense(64, dtype=tf.float64)
        self.hidden4 = layers.Dense(32, dtype=tf.float64)
        self.out = layers.Dense(1, dtype=tf.float64)

    def call(self, x):
        x = self.flatten(x)
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = keras.activations.relu(self.hidden3(x))
        x = keras.activations.relu(self.hidden4(x))
        x = self.out(x)
        return x


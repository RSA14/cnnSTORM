import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Conv2D, Flatten, Dense, Add

#
# Basic dense
densemodel = Sequential()
densemodel.add(Flatten())
densemodel.add(Dense(256, activation='tanh'))
# densemodel.add(layers.PReLU())
densemodel.add(Dense(128, activation='tanh'))
# densemodel.add(layers.PReLU())
densemodel.add(Dense(64, activation='tanh'))
# densemodel.add(layers.PReLU())
densemodel.add(Dense(32, activation='tanh'))
# densemodel.add(layers.PReLU())
densemodel.add(Dense(1))

# # Basic CNN
# model = Sequential()
# model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
# model.add(Conv2D(4, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(1))
#
# model.summary()

# 3 x Conv2D(32,3) -> BN -> PReLU -> MaxPooling2D((2,2))
conv_1 = Sequential()
conv_1.add(Conv2D(32, 3, input_shape=(32, 32, 1)))
conv_1.add(layers.BatchNormalization())
conv_1.add(layers.PReLU())
conv_1.add(layers.MaxPooling2D(pool_size=(2, 2)))

conv_1.add(Conv2D(16, 3))
conv_1.add(layers.BatchNormalization())
conv_1.add(layers.PReLU())
conv_1.add(layers.MaxPooling2D(pool_size=(2, 2)))

conv_1.add(Conv2D(8, 3))
conv_1.add(layers.BatchNormalization())
conv_1.add(layers.PReLU())
conv_1.add(layers.MaxPooling2D(pool_size=(2, 2)))
conv_1.add(layers.GlobalAveragePooling2D())
conv_1.add(Flatten())
conv_1.add(layers.Dense(1))


# General CNN generator

def create_CNN(conv, batch_norm, activation, pooling, dense, dropout,
               dense_activation=None, input_shape=(32, 32, 1), output_shape=1,
               global_avg_pool=True, filter_size=3,
               show_summary=False, plot_model=False):
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
    :param plot_model: Plots model using graphviz, good for visualisation.
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
        model = layers.Dense(dense[i], activation=dense_activation)(model)
        if dropout[i] is not None:
            model = layers.Dropout(dropout[i])(model)

    outputs = layers.Dense(output_shape)(model)

    cnn_model = keras.Model(inputs=inputs, outputs=outputs)

    if show_summary:
        cnn_model.summary()
    if plot_model:
        keras.utils.plot_model(cnn_model, show_shapes=True)

    return cnn_model


# Resnet Test
res_input = keras.Input(shape=(32, 32, 1), name="img")
x = layers.Conv2D(32, 3, activation="tanh")(res_input)
x = layers.Conv2D(64, 3, activation="tanh")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="tanh", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="tanh", padding="same")(x)
x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D(3)(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="tanh", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="tanh", padding="same")(x)
x = layers.BatchNormalization()(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="tanh")(block_3_output)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dropout(0.5)(x)
res_output = layers.Dense(1)(x)

resnet_model = keras.Model(res_input, res_output, name="toy_resnet")
# model.summary()

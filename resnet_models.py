from tensorflow import Tensor
import keras
from keras import backend as K
from keras import layers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.initializers import glorot_uniform


def initial_block(inputs, filters, activation='relu', kernel_size=5,
                  padding='valid', strides=(1, 1)):
    conv_layer = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    bn_layer = BatchNormalization()(conv_layer)

    if activation == 'relu':
        output = layers.ReLU()(bn_layer)

    elif activation == 'prelu':
        output = layers.PReLU()(bn_layer)

    else:
        return print("Unspecified activation.")

    return output


def identity_residual_block(inputs, activation='relu',
                            kernel_size=3, strides=(1, 1)):
    # Set residual and shortcut as inputs for clarity
    shortcut = inputs
    residual = inputs
    f = K.int_shape(inputs)[3]  # Filters must be fixed to input shape for dimension adding later

    # First convolution layer -> BN -> activation
    # Padding is 'same' to ensure shortcut/residual have same dims
    residual = Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same')(residual)
    residual = BatchNormalization()(residual)

    if activation == 'relu':
        residual = layers.ReLU()(residual)

    elif activation == 'prelu':
        residual = layers.PReLU()(residual)

    # Second convolution layer -> BN. Activation follows after adding shortcut
    residual = Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same')(residual)
    residual = BatchNormalization()(residual)

    add = layers.Add()([shortcut, residual])  # Add shortcut and residual, then send through activation out

    if activation == 'relu':
        output = layers.ReLU()(add)

    elif activation == 'prelu':
        output = layers.PReLU()(add)

    return output


res_input = keras.Input((32, 32, 1))
x = initial_block(res_input, filters=32, kernel_size=7, strides=(1, 1),
                  activation='relu')

x = identity_residual_block(x)
x = identity_residual_block(x)
x = identity_residual_block(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
res_output = layers.Dense(1)(x)

resnet_test = keras.Model(res_input, res_output)
# resnet_test.summary()
# keras.utils.plot_model(resnet_test, show_shapes=True)

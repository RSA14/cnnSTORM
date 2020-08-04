from tensorflow import Tensor
import keras
from keras import backend as K
from keras import layers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.initializers import glorot_uniform


def conv_bn_activation_block(inputs, filters, activation='relu', kernel_size=5,
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


def identity_block(inputs, activation='relu',
                   kernel_size=3, strides=(1, 1)):
    """
    Regular residual block, not bottlenecked.
    :param inputs:
    :param activation:
    :param kernel_size:
    :param strides:
    :return:
    """
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
    else:
        raise NotImplementedError

    # Second convolution layer -> BN. Activation follows after adding shortcut
    residual = Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same')(residual)
    residual = BatchNormalization()(residual)

    add = layers.Add()([shortcut, residual])  # Add shortcut and residual, then send through activation out

    if activation == 'relu':
        output = layers.ReLU()(add)
    elif activation == 'prelu':
        output = layers.PReLU()(add)
    else:
        raise NotImplementedError

    return output


def bottleneck_identity_block(inputs, filters=(64, 64, 256), activation='relu',
                              kernel_size=(3, 3)):
    """
    Follows the bottleneck architecture implementation of an identity block.
    :param inputs:
    :param filters:
    :param activation:
    :param kernel_size:
    :return:
    """
    # Set residual and shortcut as inputs for clarity
    shortcut = inputs
    residual = inputs
    f1, f2, f3 = filters  # Filters take specific numbers only

    # First convolution layer -> BN -> activation
    residual = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(residual)
    residual = BatchNormalization()(residual)

    if activation == 'relu':
        residual = layers.ReLU()(residual)
    elif activation == 'prelu':
        residual = layers.PReLU()(residual)
    else:
        raise NotImplementedError

    # Second convolution layer -> BN -> activation
    # Padding is 'valid' to ensure same dims
    residual = Conv2D(f2, kernel_size=kernel_size, strides=(1, 1), padding='same')(residual)
    residual = BatchNormalization()(residual)

    if activation == 'relu':
        output = layers.ReLU()(residual)
    elif activation == 'prelu':
        output = layers.PReLU()(residual)
    else:
        raise NotImplementedError

    residual = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(residual)
    residual = BatchNormalization()(residual)

    output = layers.Add()([shortcut, residual])

    if activation == 'relu':
        output = layers.ReLU()(output)
    elif activation == 'prelu':
        output = layers.PReLU()(output)
    else:
        raise NotImplementedError

    return output


def bottleneck_projection_block(inputs, filters, activation='relu',
                                kernel_size=(3, 3), strides=(2, 2)):
    # Set residual and shortcut as inputs for clarity
    shortcut = inputs
    residual = inputs
    f1, f2, f3 = filters  # Filters take specific numbers only

    # First convolution layer -> BN -> activation
    residual = Conv2D(f1, kernel_size=(1, 1), strides=strides, padding='valid')(residual)
    residual = BatchNormalization()(residual)

    if activation == 'relu':
        residual = layers.ReLU()(residual)
    elif activation == 'prelu':
        residual = layers.PReLU()(residual)
    else:
        raise NotImplementedError

    # Second convolution layer -> BN -> activation
    # Padding is 'valid' to ensure same dims
    residual = Conv2D(f2, kernel_size=kernel_size, strides=(1, 1), padding='same')(residual)
    residual = BatchNormalization()(residual)

    if activation == 'relu':
        output = layers.ReLU()(residual)
    elif activation == 'prelu':
        output = layers.PReLU()(residual)
    else:
        raise NotImplementedError

    residual = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(residual)
    residual = BatchNormalization()(residual)

    # Projection of shortcut through convolutional block with same strides as first conv layer
    shortcut = Conv2D(f3, kernel_size=(1, 1), strides=strides, padding='valid')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # Add shortcut to residual, pass through activation
    output = layers.Add()([shortcut, residual])

    if activation == 'relu':
        output = layers.ReLU()(output)
    elif activation == 'prelu':
        output = layers.PReLU()(output)
    else:
        raise NotImplementedError

    return output


def create_resnet(input_shape=(32, 32, 1), output_shape=1):
    # First convolutional layer and maxpooling
    x_input = keras.Input(input_shape)
    x = conv_bn_activation_block(x_input, filters=64, strides=(1, 1), activation='prelu', kernel_size=5)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 1
    x = bottleneck_projection_block(x, kernel_size=3, filters=[64, 64, 256], strides=1, activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[64, 64, 256], activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[64, 64, 256], activation='prelu')

    # Stage 2
    x = bottleneck_projection_block(x, kernel_size=3, filters=[128, 128, 512], strides=1, activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[128, 128, 512], activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[128, 128, 512], activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[128, 128, 512], activation='prelu')

    # Skip stage 3

    # Stage 4
    x = bottleneck_projection_block(x, kernel_size=3, filters=[64, 64, 256], activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[64, 64, 256], activation='prelu')
    x = bottleneck_identity_block(x, kernel_size=3, filters=[64, 64, 256], activation='prelu')

    # Avg pooling
    x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    ### Dense Network
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)

    # Skip connection
    _dense = layers.Dense(128)(x)
    x = x + _dense

    x = layers.Dense(64)(x)

    # Skip connection
    _dense = layers.Dense(64)(x)
    x = x + _dense

    x = layers.Dense(32)(x)

    x_output = layers.Dense(output_shape)(x)

    model = keras.Model(x_input, x_output)

    return model


resnet_model = create_resnet()
resnet_model.summary()


# keras.utils.plot_modsel(resnet_model, show_shapes=True)


# Using new architecture proposed by MSFT research team.
# Implement pre-activation using BN -> ReLU -> Conv

def preactivated_identity_residual_block(inputs, activation='relu',
                                         kernel_size=3, strides=(1, 1)):
    # Set residual and shortcut as inputs for clarity
    shortcut = inputs
    residual = inputs
    f = K.int_shape(inputs)[3]  # Filters must be fixed to input shape for dimension adding later

    # Pre-activation using BN and P/ReLU
    residual = layers.BatchNormalization()(inputs)

    if activation == 'relu':
        residual = layers.ReLU()(residual)
    elif activation == 'prelu':
        residual = layers.PReLU()(residual)
    else:
        raise NotImplementedError

    # Padding is 'same' to ensure shortcut/residual have same dims
    residual = Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same')(residual)

    residual = BatchNormalization()(residual)
    if activation == 'relu':
        residual = layers.ReLU()(residual)

    elif activation == 'prelu':
        residual = layers.PReLU()(residual)

    else:
        raise NotImplementedError

    # Second convolution layer -> addition. Only pre-activation, no post.
    residual = Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same')(residual)

    output = layers.Add()([shortcut, residual])  # Add shortcut and residual

    return output


res_input = keras.Input((32, 32, 1))
x = Conv2D(64, kernel_size=7)(res_input)
x = preactivated_identity_residual_block(x)
x = preactivated_identity_residual_block(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
res_output = layers.Dense(1)(x)

resnet_test = keras.Model(res_input, res_output)
# resnet_test.summary()
# keras.utils.plot_model(resnet_test, show_shapes=True)

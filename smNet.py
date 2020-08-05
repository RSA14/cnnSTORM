import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import resnet_models


# Define blocks using exact same architecture as smNet

def resblock(input, filters, stride=1):
    input_channels = input.shape[-1]

    residual = layers.Conv2D(filters / 4, kernel_size=3, strides=1, padding='same')(input)
    residual = layers.BatchNormalization()(residual)
    residual = layers.PReLU()(residual)

    residual = layers.Conv2D(filters / 2, kernel_size=3, strides=stride, padding='same')(residual)
    residual = layers.BatchNormalization()(residual)
    residual = layers.PReLU()(residual)

    residual = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(residual)
    residual = layers.BatchNormalization()(residual)

    if stride > 1 or input_channels != filters:
        # Perform identity shortcut convolution for dims not matching
        shortcut = layers.Conv2D(filters, strides=stride, kernel_size=1)(input)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input
    output = layers.Add()([shortcut, residual])
    output = layers.PReLU()(output)

    return output


# HardTanh activation

class HardTanh(layers.Layer):
    def __init__(self, min_z, max_z):
        super(HardTanh, self).__init__(trainable=False)
        self.min = tf.constant(min_z, dtype=tf.float32)
        self.max = tf.constant(max_z, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        if inputs < self.min:
            return self.min
        elif inputs > self.max:
            return self.max
        else:
            return inputs

    def get_config(self):
        config = {'min': float(self.min), 'max': float(self.max)}
        base_config = super(HardTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def create_smNet(input_dims=(32, 32, 1), output_dims=1, min_z=-500, max_z=500):
    input = layers.Input(shape=input_dims)

    x = layers.Conv2D(filters=64, kernel_size=7)(input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(filters=128, kernel_size=5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = resblock(x, 128, 1)
    x = resblock(x, 128, 1)
    x = resblock(x, 128, 1)

    x = resblock(x, 256, 1)

    x = resblock(x, 256, 1)
    x = resblock(x, 256, 1)
    x = resblock(x, 256, 1)

    x = layers.Conv2D(128, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(64, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(1, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    # FCs

    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(output_dims)(x)

    print(type(output))
    print(output)
    output = HardTanh(min_z=min_z, max_z=max_z)(output)
    print(type(output))
    print(output)

    model = keras.Model(input, output)

    return model

# smNet = create_smNet()
# smNet.summary()
# keras.utils.plot_model(smNet, show_shapes=True)

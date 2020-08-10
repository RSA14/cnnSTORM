import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
from keras import backend as keras_backend
from keras import layers


def MSE_loss(pred_y, y):
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

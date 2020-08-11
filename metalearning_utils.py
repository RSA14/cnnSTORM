import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
from keras import backend as keras_backend
from keras import layers
import models


def MSE_loss(pred_y, y):
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def compute_MSE_loss(model, x, y):
    pred_y = model.call(x)
    mse = MSE_loss(pred_y, y)
    return mse

def copy_model(model: keras.Model, x):
    model_copy = models.DenseModel()
    model_copy.call(x)  # To initialise weights
    model_copy.set_weights(model.get_weights())

    return model_copy

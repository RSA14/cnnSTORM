import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
import trainer
import models
import data_processing
from keras import backend as keras_backend
from keras import layers
from metalearning_utils import MSE_loss, copy_model, compute_MSE_loss


def evaluate_model_finetuned(model: keras.Model, train_data, test_data, epochs,
                             repetitions=100, batch_size=32):
    X_train, y_train = train_data
    X_test, y_test = test_data

    original_weights = model.get_weights()
    model_MSE_all = np.zeros((1, repetitions))

    for N in epochs:
        # Storage arrays for i iterations of each epoch number N
        model_MSE_epoch = np.array([])

        for i in range(repetitions):
            # Reset model weights every repetition
            model.set_weights(original_weights)

            history = trainer.train_model(model, x_train=X_train, y_train=y_train,
                                          optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                          validation_split=None, epochs=N,
                                          batch_size=batch_size, summary=False, verbose=0)

            model_MSE = model.evaluate(X_test, y_test, batch_size=batch_size)
            model_MSE_epoch = np.append(model_MSE_epoch, model_MSE)

        model_MSE_all = np.append(model_MSE_all, [model_MSE_epoch], axis=0)

    model_MSE_all = np.delete(model_MSE_all, 0, axis=0)

    return model_MSE_all


def compare_models_finetuned(models_list: [keras.Model], test_data, epochs, repetitions=100,
                             test_train_split=0.8, batch_size=32):
    print("Comparing models.")
    X_train, y_train, X_test, y_test = data_processing.split_test_train(test_data, split=test_train_split)
    all_results = []

    for i in range(len(models_list)):
        model_results = evaluate_model_finetuned(models_list[i], (X_train, y_train), (X_test, y_test),
                                                 epochs=epochs, repetitions=repetitions,
                                                 batch_size=batch_size)

        all_results.append(model_results)

    all_results = np.array(all_results)

    return all_results

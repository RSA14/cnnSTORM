import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
import keras
from keras import backend as keras_backend
from keras import layers
from metalearning_utils import MSE_loss, copy_model

def train_SOMAML(model, dataset, training_keys,
                 epochs=1, lr_inner=0.01, lr_meta=0.01,
                 inner_batch_size=32, meta_batch_size=10,
                 inner_proportion=1.0, meta_proportion=0.2):
    inner_optimizer = keras.optimizers.SGD(learning_rate=lr_inner)
    meta_optimizer = keras.optimizers.Adam(learning_rate=lr_meta)

    epoch_inner_losses = []
    epoch_meta_losses = []

    X_, y_ = dataset

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_total_inner_loss = 0
        epoch_total_meta_loss = 0

        epoch_inner_N = 0
        epoch_meta_N = 0

        for i, key in enumerate(training_keys):
            # Inner loop, optimising the learner model
            _x, _y = X_[key], y_[key]

            N = _x.shape[0]  # No. of datapoints

            indices = np.random.permutation(N)  # Shuffling training set
            inner_idx = indices[:int(np.floor(inner_proportion * N))]

            indices = np.random.permutation(N)  # Repeat for a unique testing index set, may overlap
            meta_idx = indices[:int(np.floor(meta_proportion * N))]

            inner_N, meta_N = len(inner_idx), len(meta_idx)
            epoch_inner_N += inner_N
            epoch_meta_N += meta_N

            x, y = _x[inner_idx], _y[inner_idx]
            n_batches = int(np.floor(len(x) / inner_batch_size))  # For batch processing

            model_copy = copy_model(model, x)

            # Training batches of inner loop
            batch_size = inner_batch_size
            for n in range(n_batches + 1):
                with tf.GradientTape() as train_tape:
                    if n + 1 <= n_batches:  # Check if last batch
                        inner_loss = MSE_loss(model_copy(x[0 + n * batch_size:(n + 1) * batch_size - 1]),
                                              y[0 + n * batch_size:(n + 1) * batch_size - 1])

                        epoch_total_inner_loss += inner_loss * batch_size  # Adding total loss

                    elif n + 1 > n_batches and len(x) % batch_size > 0:  # Last batch overflow
                        inner_loss = MSE_loss(model_copy(x[n * batch_size:]), y[n * batch_size:])

                        epoch_total_inner_loss += inner_loss * len(x[n * batch_size:])  # Adding total loss = n*mse

                gradients = train_tape.gradient(inner_loss, model_copy.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            # Load testing data for task T_i
            x, y = _x[meta_idx], _y[meta_idx]
            n_batches = int(np.floor(len(x) / meta_batch_size))  # For batch processing

            # Meta-training batch loop
            batch_size = meta_batch_size
            for n in range(n_batches + 1):
                with tf.GradientTape() as test_tape:
                    if n + 1 <= n_batches:  # Check if last batch
                        meta_loss = MSE_loss(model_copy(x[0 + n * batch_size:(n + 1) * batch_size - 1]),
                                             y[0 + n * batch_size:(n + 1) * batch_size - 1])

                        epoch_total_meta_loss += meta_loss * batch_size

                    elif n + 1 > n_batches and len(x) % batch_size > 0:  # Last batch overflow
                        meta_loss = MSE_loss(model_copy(x[n * batch_size:]), y[n * batch_size:])

                        epoch_total_meta_loss += meta_loss * len(x[n * batch_size:])

                gradients = test_tape.gradient(meta_loss, model_copy.trainable_variables)
                meta_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Logging MSEs for inner and meta losses per epoch
        _inner, _meta = epoch_total_inner_loss / epoch_inner_N, epoch_total_meta_loss / epoch_meta_N

        epoch_inner_losses.append(_inner)
        epoch_meta_losses.append(_meta)

        print(f"Epoch {epoch + 1} / {epochs} completed in {time.time() - epoch_start:.2f}s")
        print(f"Epoch Inner loss: {_inner}, Epoch Meta loss: {_meta}")

    plt.plot(epoch_inner_losses)
    plt.plot(epoch_meta_losses)
    plt.show()
    return epoch_inner_losses, epoch_meta_losses

    return


def train_FOMAML(model, dataset, training_keys,
                 epochs=1, lr_inner=0.01, lr_meta=0.01,
                 inner_batch_size=10, meta_batch_size=10,
                 inner_proportion=1.0, meta_proportion=0.2):
    inner_optimizer = keras.optimizers.SGD(learning_rate=lr_inner)
    meta_optimizer = keras.optimizers.Adam(learning_rate=lr_meta)

    epoch_inner_losses = []
    epoch_meta_losses = []

    X_, y_ = dataset

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_total_inner_loss = 0
        epoch_total_meta_loss = 0

        epoch_inner_N = 0
        epoch_meta_N = 0

        for i, key in enumerate(training_keys):
            # Inner loop, optimising the learner model
            _x, _y = X_[key], y_[key]

            N = _x.shape[0]  # No. of datapoints

            indices = np.random.permutation(N)  # Shuffling training set
            inner_idx = indices[:int(np.floor(inner_proportion * N))]

            indices = np.random.permutation(N)  # Repeat for a unique testing index set, may overlap
            meta_idx = indices[:int(np.floor(meta_proportion * N))]

            inner_N, meta_N = len(inner_idx), len(meta_idx)
            epoch_inner_N += inner_N
            epoch_meta_N += meta_N

            x, y = _x[inner_idx], _y[inner_idx]
            n_batches = int(np.floor(len(x) / inner_batch_size))  # For batch processing

            model_copy = keras.models.clone_model(model)
            model_copy.set_weights(model.get_weights())

            # Training batches of inner loop
            batch_size = inner_batch_size
            for n in range(n_batches + 1):
                with tf.GradientTape() as train_tape:
                    if n + 1 <= n_batches:  # Check if last batch
                        inner_loss = MSE_loss(model_copy(x[0 + n * batch_size:(n + 1) * batch_size - 1]),
                                              y[0 + n * batch_size:(n + 1) * batch_size - 1])

                        epoch_total_inner_loss += inner_loss * batch_size  # Adding total loss

                    elif n + 1 > n_batches and len(x) % batch_size > 0:  # Last batch overflow
                        inner_loss = MSE_loss(model_copy(x[n * batch_size:]), y[n * batch_size:])

                        epoch_total_inner_loss += inner_loss * len(x[n * batch_size:])  # Adding total loss = n*mse

                gradients = train_tape.gradient(inner_loss, model_copy.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            # Load testing data for task T_i
            x, y = _x[meta_idx], _y[meta_idx]
            n_batches = int(np.floor(len(x) / meta_batch_size))  # For batch processing

            # Meta-training batch loop
            batch_size = meta_batch_size
            for n in range(n_batches + 1):
                with tf.GradientTape() as test_tape:
                    if n + 1 <= n_batches:  # Check if last batch
                        meta_loss = MSE_loss(model_copy(x[0 + n * batch_size:(n + 1) * batch_size - 1]),
                                             y[0 + n * batch_size:(n + 1) * batch_size - 1])

                        epoch_total_meta_loss += meta_loss * batch_size

                    elif n + 1 > n_batches and len(x) % batch_size > 0:  # Last batch overflow
                        meta_loss = MSE_loss(model_copy(x[n * batch_size:]), y[n * batch_size:])

                        epoch_total_meta_loss += meta_loss * len(x[n * batch_size:])

                gradients = test_tape.gradient(meta_loss, model_copy.trainable_variables)
                meta_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Logging MSEs for inner and meta losses per epoch
        _inner, _meta = epoch_total_inner_loss / epoch_inner_N, epoch_total_meta_loss / epoch_meta_N

        epoch_inner_losses.append(_inner)
        epoch_meta_losses.append(_meta)

        print(f"Epoch {epoch + 1} / {epochs} completed in {time.time() - epoch_start:.2f}s")
        print(f"Epoch Inner loss: {_inner}, Epoch Meta loss: {_meta}")

    plt.plot(epoch_inner_losses)
    plt.plot(epoch_meta_losses)
    plt.show()
    return epoch_inner_losses, epoch_meta_losses


def train_REPTILE(model: keras.Model, dataset, training_keys,
                  epochs=1, inner_optimizer='SGD', lr_inner=0.01,
                  lr_meta=0.01, batch_size=10, train_proportion=1.0):
    if inner_optimizer == 'SGD':
        inner_optimizer = keras.optimizers.SGD(learning_rate=lr_inner)

    elif inner_optimizer == 'Adam':
        inner_optimizer = keras.optimizers.Adam(learning_rate=lr_inner)

    X_, y_ = dataset

    epoch_losses = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_total_loss = 0
        epoch_N = 0

        for i, key in enumerate(training_keys):
            # Inner loop for task i, SGD/Adam on the learner model
            _x, _y = X_[key], y_[key]

            N = _x.shape[0]  # No. of datapoints in task

            indices = np.random.permutation(N)  # Shuffling training set
            train_idx = indices[:int(np.floor(train_proportion * N))]

            inner_N = len(train_idx)
            epoch_N += inner_N

            x, y = _x[train_idx], _y[train_idx]
            n_batches = int(np.floor(len(x) / batch_size))  # For batch processing

            # model_copy = keras.models.clone_model(model)
            # model_copy.set_weights(model.get_weights())
            model_copy = copy_model(model, x)
            # Training batches of inner loop
            for n in range(n_batches + 1):
                with tf.GradientTape() as train_tape:
                    if n + 1 <= n_batches:  # Check if last batch
                        inner_loss = MSE_loss(model_copy(x[0 + n * batch_size:(n + 1) * batch_size - 1]),
                                              y[0 + n * batch_size:(n + 1) * batch_size - 1])

                        epoch_total_loss += inner_loss * batch_size  # Adding total loss

                    elif n + 1 > n_batches and len(x) % batch_size > 0:  # Last batch overflow
                        inner_loss = MSE_loss(model_copy(x[n * batch_size:]), y[n * batch_size:])

                        epoch_total_loss += inner_loss * len(x[n * batch_size:])  # Adding total loss = n*mse

                print(f"Loss: {inner_loss}")
                gradients = train_tape.gradient(inner_loss, model_copy.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            # Meta-update step phi <- phi + lr_meta*(phi~ - phi)
            updated_weights = []
            phi_tilde = model_copy.get_weights()
            phi = model.get_weights()

            for j in range(len(phi)):
                delta = lr_meta * (phi_tilde[j] - phi[j])
                new_weight = phi[j] + delta
                updated_weights.append(new_weight)

            model.set_weights(updated_weights)

        # Logging losses
        _loss = epoch_total_loss / epoch_N
        epoch_losses.append(_loss)

        print(f"Epoch {epoch + 1} / {epochs} completed in {time.time() - epoch_start:.2f}s")
        print(f"Epoch loss: {_loss}")

    plt.plot(epoch_losses)
    plt.show()

    return epoch_losses

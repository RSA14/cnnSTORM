import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


def train_model(model, x_train, y_train,
                optimizer='adam', loss='mse', metrics=None,
                validation_split=0.2, epochs=3, batch_size=1,
                summary=False, verbose=1
                ):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    fit = model.fit(x_train, y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, verbose=verbose)

    if summary:
        model.summary()

    return fit


def get_metrics(history, show_plot=False, plot_name=None, y_ax_lim=None):
    last_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    epoch = history.history['val_loss'].index(best_val_loss)

    records = pd.DataFrame({"last_val_loss": last_val_loss,
                            "best_val_loss": best_val_loss,
                            "best_epoch": epoch + 1}, index=[0])

    if show_plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        if y_ax_lim is not None:
            plt.ylim(y_ax_lim[0], y_ax_lim[1])
        plt.show()

        if plot_name is not None:
            plt.savefig(plot_name)

    return records


def test_model(model: keras.Model, test_set, truth, metrics=('mse')):
    metrics_dict = {}
    predictions = model.predict(test_set)

    if 'mse' in metrics:
        mse = np.mean(np.square(predictions - truth))
        metrics_dict['mse'] = mse

    return metrics_dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_model(model, x_train, y_train,
                optimizer='adam', loss='mse', metrics=['mse'],
                validation_split=0.2, epochs=3, batch_size=1,
                summary=False
                ):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    fit = model.fit(x_train, y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size)

    if summary:
        model.summary()

    return fit


def get_metrics(history, show_plot = False):
    last_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    epoch = history.history['val_loss'].index(best_val_loss)

    records = pd.DataFrame({"last_val_loss": last_val_loss,
                            "best_val_loss": best_val_loss,
                            "best_epoch": epoch}, index=[0])

    if show_plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    return records

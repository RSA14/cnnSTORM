import numpy as np
import matplotlib.pyplot as plt
from models import model
from data_generator import X_train, y_train

# Normal scaling of y_train
y_train_norm = (y_train - np.mean(y_train))/np.std(y_train)
print(np.mean(np.square(y_train_norm)))
# io.imshow(X_train[0])
# plt.show()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
print(X_train.shape)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(X_train, y_train_norm, validation_split=0.2, epochs=3)
print(model.predict(X_train[10:40]))
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





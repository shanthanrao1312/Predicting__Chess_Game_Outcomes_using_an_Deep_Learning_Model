import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sparsemax(logits):
    logits = tf.convert_to_tensor(logits)
    logits_sorted = tf.sort(logits, direction="DESCENDING")
    cumulative_sum = tf.cumsum(logits_sorted, axis=-1)
    range = tf.range(1, logits.shape[-1] + 1, dtype=logits.dtype)
    sparsemax_threshold = (cumulative_sum - 1) / tf.cast(range, logits.dtype)
    is_sparse = logits_sorted > sparsemax_threshold
    k_sparse = tf.reduce_sum(tf.cast(is_sparse, tf.int32), axis=-1)
    tau_sparse = (tf.reduce_sum(tf.where(is_sparse, logits_sorted, 0.0), axis=-1) - 1) / tf.cast(k_sparse, logits.dtype)
    return tf.maximum(logits - tau_sparse[..., tf.newaxis], 0.0)


data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels_5000.npz")
X = data['features']
y = data['labels']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(3, activation=sparsemax))


model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

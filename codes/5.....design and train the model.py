import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np


data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels.npz")


X = data['features']
y = data['labels']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential()


model.add(layers.Input(shape=(X_train.shape[1],)))


model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))


model.add(layers.Dense(3, activation='softmax'))


model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")


model.save(r"C:\Users\Asus\Desktop\Mproject\chess_mlp_model.keras")
print("Model saved successfully in Keras format.")

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import numpy as np
import tensorflow as tf


data = np.load(r"C:\Users\Asus\Desktop\Mproject\codes\code for 1000games\New folder\chess_features_labels_1000.npz")
X, y = data['features'], data['labels']


def create_model(batch_size, learning_rate, activation, dropout_rate, optimizer_name):
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(256, activation=activation, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def objective(params):
    batch_size, learning_rate, activation, dropout_rate, optimizer_name = params
    model = create_model(batch_size, learning_rate, activation, dropout_rate, optimizer_name)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=75,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=0
    )
    val_accuracy = max(history.history['val_accuracy'])
    return -val_accuracy


space = [
    Integer(32, 64, name='batch_size'),
    Real(0.0001, 0.001, name='learning_rate'),
    Categorical(['relu', 'swish'], name='activation'),
    Real(0.1, 0.4, name='dropout_rate'),
    Categorical(['adam', 'sgd'], name='optimizer')
]


res = gp_minimize(objective, space, n_calls=30, random_state=42)
print(f"Best parameters: {res.x}")
print(f"Best validation accuracy: {-res.fun}")


best_batch_size, best_learning_rate, best_activation, best_dropout_rate, best_optimizer_name = res.x
model = create_model(best_batch_size, best_learning_rate, best_activation, best_dropout_rate, best_optimizer_name)


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=75,
    batch_size=best_batch_size,
    callbacks=[early_stopping, checkpoint, lr_scheduler],
    verbose=1
)


best_model = tf.keras.models.load_model("best_model.keras")
val_accuracy = max(history.history['val_accuracy'])
print(f"Final validation accuracy: {val_accuracy}")

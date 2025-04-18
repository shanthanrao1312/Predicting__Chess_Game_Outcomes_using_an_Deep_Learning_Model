import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV  
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt


data = np.load(r"C:\Users\Asus\Desktop\Mproject\codes\5000\New folder\chess_features_labels_5000.npz")
X = data['features']
y = data['labels']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_custom_model(neurons=128, activation='relu', optimizer='adam'):
    inputs = layers.Input(shape=(X_train.shape[1],))

    x = layers.Dense(1024 if neurons == 128 else 512, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512 if neurons == 128 else 256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(0.4)(x)

    residual = x
    x = layers.Dense(512 if neurons == 128 else 256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.add([x, residual])

    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(neurons, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_custom_model, verbose=0)

param_grid = {
    'batch_size': [32, 64],
    'epochs': [20, 50],
    'model__activation': ['relu', 'swish'],
    'model__neurons': [128, 256],  
    'optimizer': ['adam', 'sgd']
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)

grid_result = grid.fit(X_train, y_train)

print("All results from grid search:")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print(f"Mean Accuracy: {mean:.4f}, Std: {std:.4f} with: {param}")

print("\nBest Accuracy:")
print(f"Accuracy: {grid_result.best_score_:.4f}")
print(f"With Parameters: {grid_result.best_params_}")

best_model = grid_result.best_estimator_.model_
best_model.save(r"C:\Users\Asus\Desktop\Mproject\chess_mlp_model_5000_optimized.keras")
print("Best model saved successfully.")

test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy on unseen data: {test_accuracy:.4f}")

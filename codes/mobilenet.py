import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = np.load(r"C:\Users\Asus\Desktop\Mproject\codes\code for 1000games\New folder\chess_features_labels_1000.npz")
X = data['features']
y = data['labels']

num_classes = len(np.unique(y))

y = tf.keras.utils.to_categorical(y, num_classes)

def data_generator(X, y, batch_size=32):
    data_size = X.shape[0]
    while True:
        for i in range(0, data_size, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            # Resize and convert to RGB
            X_batch_resized = np.array([
                tf.image.resize(tf.image.grayscale_to_rgb(tf.expand_dims(x, axis=-1)), (224, 224)).numpy()
                for x in X_batch
            ]) / 255.0
            yield X_batch_resized, y_batch

batch_size = 32
train_gen = data_generator(X, y, batch_size=batch_size)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')

steps_per_epoch = X.shape[0] // batch_size

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=(X_resized, y),
    callbacks=[early_stopping, checkpoint]
)

test_loss, test_accuracy = model.evaluate(train_gen, steps=steps_per_epoch)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

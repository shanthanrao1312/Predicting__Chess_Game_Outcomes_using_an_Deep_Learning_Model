import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

data = np.load(r"C:\Users\Asus\Desktop\Mproject\codes\5000\chess_features_labels_5000.npz")
X = data['features']
y = data['labels']
input_shape = X.shape[1]
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

def residual_block(x, units):
    shortcut = x
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def create_resnet(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    x = Dense(128)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = create_resnet(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, validation_split=0.2, epochs=10, batch_size=32)

test_loss, test_accuracy = model.evaluate(X, y)
print(f'Test Accuracy: {test_accuracy}')

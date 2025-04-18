import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

data = np.load(r"C:\Users\Asus\Desktop\Mproject\codes\code for 10games\New folder\chess_features_labels.npz")
X = data['features']
y = data['labels']

print(f"NaNs in features: {np.isnan(X).sum()}, NaNs in labels: {np.isnan(y).sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_classes = len(np.unique(y))  
y_categorical = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

def create_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),  
        Dense(320, activation='relu'),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(X_train.shape[1], num_classes)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")



try:
    loaded_model = load_model(r"C:\Users\Asus\Desktop\Mproject\codes\5000\best_model.h5")
    loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_test_accuracy}")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")

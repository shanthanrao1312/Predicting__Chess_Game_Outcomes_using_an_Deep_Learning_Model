import numpy as np
from sklearn.model_selection import train_test_split

data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels.npz")

X = np.random.rand(100, 73)
y = np.random.randint(0, 3, 100)


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)


print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)


import numpy as np
from tensorflow.keras.models import load_model


model = load_model(r"C:\Users\Asus\Desktop\Mproject\chess_mlp_model.keras")


new_data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels_next_10_games.npz")

X_new = new_data['features']
y_new = new_data['labels']

game_ids = new_data['games']

predictions = model.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)

current_game = game_ids[0]
print(f"--- Predictions for Game {current_game} ---")


for i in range(len(predicted_classes)):
    if game_ids[i] != current_game:
        current_game = game_ids[i]
        print(f"\n--- Predictions for Game {current_game} ---")

    print(f"Position {i + 1}: Predicted = {predicted_classes[i]}, Actual = {y_new[i]}")


accuracy = np.mean(predicted_classes == y_new)
print(f"\nOverall Accuracy on new 10 games: {accuracy:.4f}")

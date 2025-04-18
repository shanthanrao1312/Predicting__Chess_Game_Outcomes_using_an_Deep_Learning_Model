import numpy as np


stockfish_data = np.load(r"C:\Users\Asus\Desktop\Mproject\stockfish_evaluations_5000.npz", allow_pickle=True)
stockfish_evaluations = stockfish_data['evaluations']
stockfish_fens = stockfish_data['fens']


mlp_data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels_fensadd_fen_5000.npz", allow_pickle=True)
mlp_predictions = mlp_data['labels']  
mlp_fens = mlp_data['fens']

if not np.array_equal(stockfish_fens, mlp_fens):
    print("Mismatch in FEN positions between Stockfish and MLP data.")
else:
    print("FEN positions match between Stockfish and MLP data.")

print("Comparing Stockfish evaluations and MLP predictions:")
for i, (sf_eval, mlp_pred, fen) in enumerate(zip(stockfish_evaluations, mlp_predictions, stockfish_fens)):
    print(f"Position {i + 1} (FEN: {fen}):")
    print(f"  Stockfish Evaluation: {sf_eval}")
    print(f"  MLP Prediction: {mlp_pred}")
    print(f"  Difference: {abs(sf_eval - mlp_pred)}")
    print()

differences = np.abs(stockfish_evaluations - mlp_predictions)
print(f"Average difference between Stockfish and MLP evaluations: {np.mean(differences):.2f}")
print(f"Maximum difference between Stockfish and MLP evaluations: {np.max(differences)}")

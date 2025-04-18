import chess
import chess.engine
import numpy as np
import random

data = np.load(r"C:\Users\Asus\Desktop\Mproject\stockfish_evaluations_5000.npz", allow_pickle=True)
stored_evaluations = data['evaluations']
fens = data['fens']

engine_path = r"C:\Users\Asus\Desktop\Mproject\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

num_samples = 10
sample_indices = random.sample(range(len(fens)), num_samples)
differences = []

for idx in sample_indices:
    fen = fens[idx]
    stored_eval = stored_evaluations[idx]

    board = chess.Board(fen)
    
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    new_eval = info['score'].relative.score()

    if new_eval is None:
        new_eval = 0
    elif info['score'].relative.is_mate():
        new_eval = 10000 if info['score'].relative.mate() > 0 else -10000

    difference = abs(stored_eval - new_eval)
    differences.append(difference)

    print(f"Position {idx + 1} (FEN: {fen}):")
    print(f"  Stored Evaluation: {stored_eval}")
    print(f"  New Evaluation: {new_eval}")
    print(f"  Difference: {difference}")
    print("\n")

avg_difference = sum(differences) / len(differences)
print(f"Average difference in the sampled evaluations: {avg_difference}")
engine.quit()

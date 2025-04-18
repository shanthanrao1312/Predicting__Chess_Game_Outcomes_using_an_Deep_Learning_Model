import chess
import chess.engine
import numpy as np
import re

data = np.load(r"C:\Users\Asus\Desktop\Mproject\chess_features_labels_fens_5000.npz", allow_pickle=True)
fens = data['fens']
features = data['features']
labels = data['labels']

engine_path = r"C:\Users\Asus\Desktop\Mproject\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

stockfish_evaluations = []

def is_valid_fen(fen):
    if not isinstance(fen, str):
        return False
    return re.match(r"^([pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+ [wb] (K?Q?k?q?|-) ([a-h][36]|-) \d+ \d+$", fen)

for i, fen in enumerate(fens):
    try:
        fen = str(fen)

        if not is_valid_fen(fen):
            print(f"Skipping invalid FEN at position {i + 1}: {fen}")
            continue

        board = chess.Board(fen)

        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        evaluation = info['score'].relative.score()

        if evaluation is None:
            evaluation = 0
        elif info['score'].relative.is_mate():
            evaluation = 10000 if info['score'].relative.mate() > 0 else -10000

        stockfish_evaluations.append(evaluation)
        print(f"Evaluating position {i + 1}/{len(fens)}: Stockfish evaluation = {evaluation}")

    except Exception as e:
        print(f"Error at position {i + 1}: {e}")

engine.quit()

output_path = r"C:\Users\Asus\Desktop\Mproject\chess_features_labels_fens_5000.npz"
np.savez(output_path, features=features, labels=labels, fens=fens, evaluations=stockfish_evaluations)
print(f"Stockfish evaluations saved to {output_path}")

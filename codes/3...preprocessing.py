import chess.pgn
import numpy as np

def parse_fen(fen):
    board, active_color, castling, en_passant, halfmove_clock, fullmove_number = fen.split()


    piece_map = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        'P': 1,  'N': 2,  'B': 3,  'R': 4,  'Q': 5,  'K': 6,
    }

    board_vector = []
    for row in board.split('/'):
        for char in row:
            if char.isdigit():
                board_vector.extend([0] * int(char))
            else:
                board_vector.append(piece_map[char])


    active_color_vector = 0 if active_color == 'w' else 1


    castling_vector = [
        1 if 'K' in castling else 0,
        1 if 'Q' in castling else 0,
        1 if 'k' in castling else 0,
        1 if 'q' in castling else 0,
    ]


    en_passant_vector = [0, 0]
    if en_passant != '-':
        en_passant_vector = [1, chess.SQUARES[chess.parse_square(en_passant)]]


    halfmove_clock = int(halfmove_clock)
    fullmove_number = int(fullmove_number)


    features = board_vector + [active_color_vector] + castling_vector + en_passant_vector + [halfmove_clock, fullmove_number]

    return np.array(features)

def process_pgn_file(pgn_file_path):
    all_features = []
    all_labels = []

    with open(pgn_file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            result = game.headers['Result']
            if result == "1-0":
                label = 0
            elif result == "0-1":
                label = 1
            else:
                label = 2

            board = game.board()

            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                features = parse_fen(fen)
                all_features.append(features)
                all_labels.append(label)

    return np.array(all_features), np.array(all_labels)


pgn_file_path = r"C:\Users\Asus\Desktop\Mproject\extracted_10_games.pgn"


all_features, all_labels = process_pgn_file(pgn_file_path)


save_path = r"C:\Users\Asus\Desktop\Mproject\chess_features_labels.npz"
np.savez(save_path, features=all_features, labels=all_labels)

print(f"Saved {len(all_features)} feature vectors and labels to {save_path}")

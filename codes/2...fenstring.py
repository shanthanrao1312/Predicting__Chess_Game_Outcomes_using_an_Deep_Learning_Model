import chess.pgn


pgn_file = r"C:\Users\Asus\Desktop\Mproject\extracted_10_games.pgn"


with open(pgn_file) as pgn:
    game_number = 1
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        print(f"Game {game_number} between {game.headers['White']} and {game.headers['Black']}")
        print(f"Result: {game.headers['Result']}")

        board = game.board()
        fen_strings = []

        for move in game.mainline_moves():
            board.push(move)
            fen = board.fen()
            fen_strings.append(fen)


        for i, fen in enumerate(fen_strings):
            print(f"Move {i + 1}: {fen}")

        game_number += 1
        print("\n")


import chess.pgn

def extract_and_view_10_games(input_pgn_path, output_pgn_path):

    with open(input_pgn_path) as pgn_file:
        with open(output_pgn_path, 'w') as out_file:
            for i in range(10):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  
                game_str = str(game)
                out_file.write(game_str + "\n\n")
                print(f"Game {i + 1}:\n{game_str}\n")
    print(f"Extracted and displayed 10 games to {output_pgn_path}")

input_pgn = r"C:\Users\Asus\Desktop\Mproject\DATABASE4U.pgn"
output_pgn = r"C:\Users\Asus\Desktop\Mproject\extracted_10_games.pgn"

extract_and_view_10_games(input_pgn, output_pgn)
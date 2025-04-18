import chess.pgn

def extract_and_view_next_10_games(input_pgn_path, output_pgn_path, start_game=11, num_games=10):

    with open(input_pgn_path) as pgn_file:

        for _ in range(start_game - 1):
            chess.pgn.read_game(pgn_file)


        with open(output_pgn_path, 'w') as out_file:
            for i in range(num_games):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                game_str = str(game)
                out_file.write(game_str + "\n\n")
                print(f"Game {start_game + i}:\n{game_str}\n")
    print(f"Extracted and displayed games {start_game} to {start_game + num_games - 1} to {output_pgn_path}")


input_pgn = r"C:\Users\Asus\Desktop\Mproject\DATABASE4U.pgn"
output_pgn = r"C:\Users\Asus\Desktop\Mproject\extracted_next_10_games.pgn"

extract_and_view_next_10_games(input_pgn, output_pgn, start_game=11, num_games=10)

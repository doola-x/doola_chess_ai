import chess
import chess.engine
import subprocess
import random

def append_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

def main():
	engine_path = "../stockfish"
	game_data = []
	discount = 0.91
	move_no = 0
	with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
		while True:
			board = chess.Board()
			if (random.random() > 0.1):
				moves = [board.san(move) for move in board.legal_moves]
				rand_no = round(random.random() * len(moves))
				if rand_no != 0:
					rand_no = rand_no - 1
				rand = moves[rand_no]
				board.push_san(rand)
				print(f"pushed random move: {rand}")

			while not board.is_game_over():
				if board.turn == chess.BLACK:
					# Assuming the human or your policy network makes a move
						result = subprocess.run(f'/opt/homebrew/bin/python3.11 inference.py "{board.fen()}"', capture_output=True, shell=True)
						print(f"result: {result}")
						if result.stdout:
							move = result.stdout.decode().strip()
						board.push_san(move)  # Assumes move is given in standard algebraic notation
				else:
					# Let Stockfish make a move as White
					result = engine.play(board, chess.engine.Limit(time=0.1))  # Play with a time limit of 0.1 seconds
					board.push(result.move)
					print(f"Stockfish plays: {result.move}")
				game_data.append((board.fen(), move_no))
				move_no += 1
				print(board)
			outcome_value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
			outcome = board.result()

			for state, move in game_data:
				discount_exp = len(game_data) - move - 1
				discounted_outcome = (discount ** discount_exp) * outcome_value[outcome]
				discounted_outcome = round(discounted_outcome, 2)
				print(f"state: {state}")
				print(f"discounted_outcome: {discounted_outcome}")
				append_to_file('../data/value_training/stockfish_white.txt', f"{state}:{discounted_outcome}")
			
			print("Game over")
			print("Result:", outcome)

if __name__ == "__main__":
	main()
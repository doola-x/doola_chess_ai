import os
import io
import gzip
import json
import torch
import chess
import chess.pgn

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

move_dict = '../data/moves0.json'
with open(move_dict, 'r') as file:
    move_dict = json.load(file)

def fen_to_tensor(fen):
    board_tensor = torch.zeros(8, 8, 13)
    
    pieces, active_color, _, _, _, _ = fen.split(' ')
    rows = pieces.split('/')

    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                piece_idx = piece_to_idx[char]
                board_tensor[i, col, piece_idx] = 1
                col += 1
                
    if active_color == 'w':
        board_tensor[:, :, 12] = 1  # Indicate active color is white
    else:
        board_tensor[:, :, 12] = 0  # Indicate active color is black
    print(f"board: {board_tensor}")
    return board_tensor

def process_files(directory): 
	game_c = 1
	for filename in os.listdir(directory):
		filepath = os.path.join(directory, filename)
		print(filepath)
		with open(filepath, 'r') as file:
			content = file.read()

		games = content.strip().split('\n')
		for game in games:
			if (len(game) > 0 and game[0] == '1'):
				#print(f"game: {game}\n")
				pgn = io.StringIO(game)
				game = chess.pgn.read_game(pgn)
				board = game.board()
				move_c = 0
				for move in game.mainline_moves():
					tensor = fen_to_tensor(board.fen())
					board.push(move)
					print(f"move: {move_dict[move.uci()]}, fen: {board.fen()}")
					with gzip.open(f"../data/processed_0/{game_c}/move{move_c}.gz", 'wb') as f:
						torch.save({'state': tensor, 'action': move_dict[move.uci()]}, f)
					move_c += 1

				game_c += 1

process_files("../data/raw_data.0/")
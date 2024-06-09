import chess
import torch
import argparse
import numpy as np

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

def tensor_to_board(tensor):
    board = chess.Board(None)  # Start with an empty board
    piece_map = {0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K', 6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'}
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            piece = tensor[i, j].argmax()
            if piece.item() > 0:  # Assuming 0 is empty square
                square = chess.square(j, 7 - i)  # Convert tensor index to chess square
                board.set_piece_at(square, chess.Piece(piece.item(), chess.WHITE if piece.item() <= 5 else chess.BLACK))
    return board.fen()

def fen_to_tensor(fen):
    board_tensor = np.zeros((8, 8, 13), dtype=np.float32)
    
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
    
    return torch.tensor(board_tensor)

def main(fen):
	# Example use case
	board = fen_to_tensor(fen)
	print(board)
	new_fen = tensor_to_board(board)
	print(f"FEN: {new_fen}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next chess move from a given board state.")
    parser.add_argument("fen", help="FEN string representing the board state")
    args = parser.parse_args()
    
    main(args.fen)

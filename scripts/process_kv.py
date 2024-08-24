import chess
import os
import torch
import numpy as np

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

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

def read_and_process_files(directory):
    count = 1
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(filepath)
            
            with open(filepath, 'r') as file:
                content = file.read()

            # Split games based on two blank lines
            tactics = content.split('\n')

            # Process each game
            for tactic in tactics:
                parts = tactic.split(':')
                #print(parts[0] + " <- fen move ->" + parts[1])
                tensor = fen_to_tensor(parts[0])
                float_val = (parts[1])
                np.savez_compressed(f'../data/processed_value/{count}.npz', state=tensor, correct_move=float_val)
                count+=1

read_and_process_files("../data/value_training/")


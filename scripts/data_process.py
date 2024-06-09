import chess
import os
import re
import json
import torch
import numpy as np

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

def parse_pgn(raw_pgn, count):
    moves = extract_moves(raw_pgn)
    processed_moves = []
    move_no = 1

    board = chess.Board()
    previous_state = -1
    for i in range(0, len(moves)):
        try:
            #push new move to check if legal
            board.push_san(moves[i])

            if (previous_state != -1):
                #if previous state is set, that is the input state
                tensor = fen_to_tensor(previous_state)
            else:
                #if start of game, 0s tensor
                tensor = torch.zeros((8, 8, 13))

            #set key and previous state for next iteration
            key = moves[i]
            previous_state = board.fen()

            np.savez_compressed(f'../data/processed_games_2/{count}/{move_no}.npz', state=tensor, correct_move=key, fen=previous_state)
        except Exception as error:
            print("illegal!")
            print(error)
            #do nothing
            pass
        move_no+=1

def extract_moves(raw_pgn):
    start_index = raw_pgn.find("\n\n")
    if start_index == -1:
        return []

    end_markers = [" 1-0", " 0-1", " 1/2-1/2"]
    end_index = len(raw_pgn)
    for marker in end_markers:
        index = raw_pgn.find(marker, start_index)
        if index != -1:
            end_index = index

    moves_text = raw_pgn[start_index:end_index].strip()
    moves = moves_text.split()

    filtered_moves = []
    for move in moves:
        if move[0] in 'abcdefghKQRBNO': 
            filtered_moves.append(move)
    return filtered_moves

# Function to convert FEN to a tensor
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
    count=1
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(filepath)
            
            with open(filepath, 'r') as file:
                content = file.read()

            # Split games based on two blank lines
            games = content.strip().split('\n\n\n')

            # Process each game
            for game in games:
                parse_pgn(game, count)
                count+=1
                


directory = "../data/raw_data/"  # Change to your directory path
read_and_process_files(directory)


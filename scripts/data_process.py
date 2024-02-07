import chess
import os
import re
import json
import numpy as np

allgames = []
allcolors = []

def parse_pgn(raw_pgn):
    # Initialize a chess board
    board = chess.Board()
    white = re.search(r'\[White "(.*?)"\]', raw_pgn)
    black = re.search(r'\[Black "(.*?)"\]', raw_pgn)

    # Determine your color
    color = None

    if white and white.group(1) == "doolasux":
        color = 'white'
    elif black and black.group(1) == "doolasux":
        color = 'black'

    # Extract moves using regex or a chess library
    moves = extract_moves(raw_pgn)
    print(moves)
    # Process each move
    processed_moves = []
    for move in moves:
        try:
            board.push_san(move)
            # Convert the current board state to a numerical format
            board_state = board_to_numerical(board)
            processed_moves.append(board_state)
        except:
            #do nothing
            pass
    return processed_moves, color

def extract_moves(raw_pgn):
    # Regex pattern to match SAN moves, ignoring annotations and metadata
    start_index = raw_pgn.find("\n\n")
    if start_index == -1:
        return []

    # Find the end of the moves (e.g., using game result as marker)
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
        if move[0].isdigit() and '.' in move:  # Check if it's a move number
            filtered_moves.append(move)
        elif move[0] in 'abcdefghKQRBN':  # Check if it's a move
            filtered_moves.append(move)

    return extract_san_moves(filtered_moves)


def is_valid_san(move):
    if '}' in move:
        return False
    # Check for castling moves
    if move in ['O-O', 'O-O-O']:
        return True

    # Check for normal moves (like 'e4', 'Nf3')
    if len(move) >= 2 and move[0] in 'abcdefghNBRQK':
        if move[-1] in '12345678+#':
            return True

    # Check for captures (like 'exd5')
    if 'x' in move:
        if len(move) >= 3 and move[0] in 'abcdefghNBRQK':
            if move[-1] in '12345678+#':
                return True

    return False

def extract_san_moves(elements):
    combined_moves = []
    for el in elements:
        if is_valid_san(el):
            combined_moves.append(el)

    return combined_moves


def board_to_numerical(board):
    # Define a mapping for pieces to indices in the one-hot vector
    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Initialize a blank board (8x8x13)
    numerical_board = np.zeros((8, 8, 13))
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:  # If there's a piece on the square
            # Determine the index for one-hot vector
            piece_idx = piece_to_index[piece.piece_type]
            color_idx = 0 if piece.color == chess.WHITE else 6

            # Set the appropriate slot in the vector
            numerical_board[chess.square_rank(square)][chess.square_file(square)][piece_idx + color_idx] = 1
        else:
            # Mark the square as empty
            numerical_board[chess.square_rank(square)][chess.square_file(square)][12] = 1

    return numerical_board

def read_and_process_files(directory):
    processed_games = []
    
    # List all PGN files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(filepath)
            
            # Read the content of the file
            with open(filepath, 'r') as file:
                content = file.read()

            # Split games based on two blank lines
            games = content.strip().split('\n\n\n')

            # Process each game
            count = 0
            for game in games:
                processed_game, color = parse_pgn(game)
                allgames.append(processed_game)
                np.savez_compressed(f'processed_games_{filename}.npz', states=processed_game)


directory = "../data/raw_data/"  # Change to your directory path
read_and_process_files(directory)


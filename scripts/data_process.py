import chess
import os
import re
import numpy as np

def parse_pgn(raw_pgn):
    # Initialize a chess board
    board = chess.Board()

    # Extract moves using regex or a chess library
    moves = extract_moves(raw_pgn)

    # Process each move
    processed_moves = []
    print('start of moves')
    for move in moves:
        try:
            print(move)
            board.push_san(move)
            # Convert the current board state to a numerical format
            board_state = board_to_numerical(board)
            processed_moves.append(board_state)
        except:
            #do nothing
            pass
    return processed_moves

def extract_moves(raw_pgn):
    # Regex pattern to match SAN moves, ignoring annotations and metadata
    move_pattern = r"\b(?:[a-hKQRBN][a-h1-8]?x?[a-h1-8](?:=[QRBN])?|O-O-O|O-O)\b"
    moves = re.findall(move_pattern, raw_pgn, re.MULTILINE)

    return moves

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
            
            # Read the content of the file
            with open(filepath, 'r') as file:
                content = file.read()

            # Split games based on two blank lines
            games = content.strip().split('\n\n\n')

            # Process each game
            for game in games:
                processed_game = parse_pgn(game)
                append_to_txt("processed_games.txt", processed_game)

def append_to_txt(file_name, data):
    with open(file_name, 'a') as file:  # 'a' for append mode
        count = 0
        for item in data:
            # Format the data as a string
            formatted_data = str(item) + '\n'
            file.write(formatted_data)
            count+=1
            print(count)


directory = "../data/raw_data"  # Change to your directory path
all_processed_games = read_and_process_files(directory)

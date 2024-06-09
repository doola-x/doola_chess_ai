import chess
import pandas as pd

file_path = "../data/tactics/lichess_puzzle/round1.txt"

def append_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

# Load CSV into DataFrame
df = pd.read_csv('../data/lichess_db_puzzle.csv')

for index, row in df.iterrows():
    #print(row['FEN'], row['Moves'])
    #for each fen, load up a chess board to translate moves
    fen = row['FEN']
    moves = row['Moves'].split(' ')
    count = 0
    for move in moves:
        board = chess.Board(fen)
        print(str(count) + ":" + move)
        move_obj = board.push_san(move)
        at_sqr = move_obj.from_square
        to_sqr = move_obj.to_square
        board.pop()
        piece = board.piece_at(at_sqr)
        capture = board.piece_at(to_sqr)

        if capture != None:
            cap = True
            takes = 'x'
        else:
            cap = False
            takes = ''
            
        if (piece.symbol() == 'P' or piece.symbol() == 'p'):
            if cap == True:
                symbol = chess.SQUARE_NAMES[move_obj.from_square]
                symbol = symbol[0]
                print('in pawn takes, col x col: ' + symbol)
            else:
                symbol = ''
        else:
            symbol = piece.symbol().upper()

        formatted_move = f'{fen};{symbol}{takes}{chess.SQUARE_NAMES[move_obj.to_square]}'
        if count != 0:
            append_to_file(file_path, formatted_move)
        count+=1
        board.push_san(move)
        fen = board.fen()
#176 possible pawn moves (not including promotions), 128 moves for each remaining piece (no need to include check information? chess engine can handle that)
#640 possible piece moves = 816 possilbe moves
import json

def generate_move_mapping():
    files = 'abcdefgh'
    ranks = '12345678'
    pieces = ' NBRQK'
    moves = {}
    move_id = 0

    # Generate regular moves and captures for non-pawn pieces
    for piece in pieces:
        for end_file in files:
            for end_rank in ranks:
                #if pawn, piece is null
                if piece == " ":
                    piece = ""
                #get neighbor ranks
                move = f"{piece}{end_file}{end_rank}"
                moves[move] = move_id
                move_id += 1

                if piece == "":
                    #if piece is a pawn, capture specify starting file
                    idx = files.find(end_file)
                    neighbor_l = idx - 1
                    neighbor_r = idx + 1
                    if neighbor_l >= 0:
                        neighbor_l = files[neighbor_l]
                        capture_move_l = f"{end_file}x{neighbor_l}{end_rank}"
                        moves[capture_move_l] = move_id
                        move_id += 1
                    if neighbor_r < 8:
                        neighbor_r = files[neighbor_r]
                        capture_move_r = f"{end_file}x{neighbor_r}{end_rank}"
                        moves[capture_move_r] = move_id
                        move_id += 1
                else:
                    #file specific moves, like Rae1
                    if (piece != "K"):
                        for specific in files:
                            move = f"{piece}{specific}{end_file}{end_rank}"
                            moves[move] = move_id
                            move_id += 1
                    if (piece != "K"):
                        for specific in ranks:
                            move = f"{piece}{specific}{end_file}{end_rank}"
                            moves[move] = move_id
                            move_id += 1
                    capture_move = f"{piece}x{end_file}{end_rank}"
                    moves[capture_move] = move_id
                    move_id += 1
                    if (piece != "K"):
                        for specific in files:
                            move = f"{piece}{specific}x{end_file}{end_rank}"
                            moves[move] = move_id
                            move_id += 1
                    if (piece != "K"):
                        for specific in ranks:
                            move = f"{piece}{specific}x{end_file}{end_rank}"
                            moves[move] = move_id
                            move_id += 1

    return moves

move_mapping = generate_move_mapping()
with open("moves.json", "w") as f:
    f.write(json.dumps(move_mapping))
print(f"Total moves mapped: {len(move_mapping)}")
import itertools
import json
import os

def generate_chess_move_mappings_json(output_file="moves0.json"):
    # Generate all squares on the board (a1 to h8)
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f + r for f, r in itertools.product(files, ranks)]

    # Generate all possible move notations (source -> destination)
    move_mappings = {}
    index = 0

    promotion_ranks = {"white": "8", "black": "1"}
    promotion_pieces = "qrbn"  # Queen, Rook, Bishop, Knight

    for source, destination in itertools.product(squares, repeat=2):
        if source != destination:  # Exclude moves from a square to itself
            source_file, source_rank = source[0], source[1]
            destination_file, destination_rank = destination[0], destination[1]

            # Regular move
            notation = f"{source}{destination}"
            move_mappings[notation] = index
            index += 1

            # Add promotion notations if applicable
            if source_rank == "7" and destination_rank == promotion_ranks["white"]:  # White promotion
                if source_file == destination_file or abs(ord(source_file) - ord(destination_file)) == 1:
                    for piece in promotion_pieces:
                        promotion_notation = f"{notation}{piece}"
                        move_mappings[promotion_notation] = index
                        index += 1
            elif source_rank == "2" and destination_rank == promotion_ranks["black"]:  # Black promotion
                if source_file == destination_file or abs(ord(source_file) - ord(destination_file)) == 1:
                    for piece in promotion_pieces:
                        promotion_notation = f"{notation}{piece}"
                        move_mappings[promotion_notation] = index
                        index += 1

    # Append or create the JSON file
    if os.path.exists(output_file):
        # If file exists, load existing data
        with open(output_file, "r") as file:
            existing_data = json.load(file)
    else:
        # If file does not exist, initialize an empty dictionary
        existing_data = {}

    # Merge new data with existing data
    existing_data.update(move_mappings)

    # Save back to the JSON file
    with open(output_file, "w") as file:
        json.dump(existing_data, file, indent=4)

    print(f"Generated {len(move_mappings)} move mappings and appended to '{output_file}'.")

if __name__ == "__main__":
    generate_chess_move_mappings_json()

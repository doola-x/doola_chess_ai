import argparse
import torch
import torch.nn as nn
import numpy as np
import json


class ChessModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def decode_move(value, filename='../data/moves.json'):
    """Decodes a move from the model's output integer back to SAN notation by loading a mapping from a file."""
    try:
        with open(filename, 'r') as file:
            mapping = json.load(file)
            # Reverse the mapping to find the key by value
            for key, val in mapping.items():
                if val == value:
                    return key
            # If no matching value is found
            raise ValueError(f"No move found for value {value}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found.")

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

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

def main(fen):
    model = ChessModel(input_size=832, output_size=9010) 

    # Load the model weights
    model.load_state_dict(torch.load('model_epoch_50.pth'))
    model.eval()

    tensor = fen_to_tensor(fen)

    with torch.no_grad():
        tensor_flattened = tensor.view(-1)  # This correctly flattens the entire tensor
        tensor_flattened = tensor_flattened.unsqueeze(0) 
        prediction = model(tensor_flattened.float())
        predicted_move = prediction.argmax(dim=1)

        # Convert the predicted move from its encoded format back to a standard move format
        decoded_move = decode_move(predicted_move.item())
        print(f"Suggested move: {decoded_move}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next chess move from a given board state.")
    parser.add_argument("fen", help="FEN string representing the board state")
    args = parser.parse_args()
    
    main(args.fen)
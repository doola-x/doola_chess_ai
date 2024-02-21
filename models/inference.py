import argparse
import torch
import torch.nn as nn
import numpy as np
import json


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Assuming the output of conv layers is flattened, adjust the input size accordingly
        # LSTM input dimensions: (batch_size, seq_len, features)
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 9010)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        # Flatten the output for the LSTM
        x = self.flatten(x)
        # Reshape x to have a sequence length of 1, as each board state is independent
        x = x.unsqueeze(1)  # batch_size x 1 x (64*8*8)
        lstm_out, (hn, cn) = self.lstm(x)
        # Only use the last hidden state
        x = self.fc(lstm_out[:, -1, :])
        return x
model = ChessModel()

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
    model = ChessModel() 

    # Load the model weights
    model.load_state_dict(torch.load('model_epoch_31.pth'))
    model.eval()

    tensor = fen_to_tensor(fen)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)
        predicted_move = prediction.argmax(dim=1)

        # Convert the predicted move from its encoded format back to a standard move format
        decoded_move = decode_move(predicted_move.item())
        print(f"Suggested move: {decoded_move}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next chess move from a given board state.")
    parser.add_argument("fen", help="FEN string representing the board state")
    args = parser.parse_args()
    
    main(args.fen)
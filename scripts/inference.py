import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import chess

move_dict = '../data/moves.json'

with open(move_dict, 'r') as file:
    move_dict = json.load(file)


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Assuming the output of conv layers is flattened, adjust the input size accordingly
        # LSTM input dimensions: (batch_size, seq_len, features)
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc = nn.Linear(1024, 9010)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjusting dimensions for the Conv2D layer
        x = self.relu(self.conv1(x))  # Pass through the first convolutional layer and ReLU
        x = self.flatten(x)  # Flatten the output to feed into LSTM
        x = x.unsqueeze(1)  # Adjusting the shape for LSTM (batch_size, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Using the last layer's hidden state. You mentioned using lstm_out, but usually hn[-1] is used
        # Using lstm_out's last sequence output if the LSTM's `batch_first=True`
        x = hn[-1]  # If you're sure about using lstm_out, replace `hn[-1]` with `lstm_out[:, -1, :]`
        x = self.fc(x)
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

def adjust_logits(logits, legal_moves_masks):
    """
    Adjusts logits for a batch of game states, penalizing illegal moves.
    
    :param logits: A 2D tensor of logits from the model (batch_size x num_moves).
    :param legal_moves_masks: A boolean tensor indicating legal moves (batch_size x num_moves).
    :return: Adjusted logits.
    """
    illegal_moves_penalty = -1e9
    inverse_mask = 1 - mask  # This subtracts each element in mask from 1, effectively inverting it
    adjusted_logits = logits + (inverse_mask.float() * illegal_moves_penalty)   
    return adjusted_logits

def main(fen):
    model = ChessModel() 

    # Load the model weights
    model.load_state_dict(torch.load('../models/model_epoch_15.pth'))
    model.eval()

    tensor = fen_to_tensor(fen)
    tensor = tensor.unsqueeze(0)
    move_val = float('-inf')
    output = ''
    with torch.no_grad():
        prediction = model(tensor)
        for row in prediction:
            for i in range(len(row)):
                if (row[i] > move_val):
                    board = chess.Board(fen)
                    try:
                        new_move = decode_move(i)
                        board.push_san(new_move)
                        #doesnt except, legal move
                        move_val = row[i]
                        #print(f'new legal move suggestion: {new_move}')
                        output = new_move
                    except:
                        #print(f'illegal move suggestion!')
                        j = 0


        #predicted_move = prediction.argmax(dim=1)

        # Convert the predicted move from its encoded format back to a standard move format
        #decoded_move = decode_move(predicted_move.item())
        print(f"{output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next chess move from a given board state.")
    parser.add_argument("fen", help="FEN string representing the board state")
    args = parser.parse_args()
    
    main(args.fen)
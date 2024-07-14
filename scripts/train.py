import sys
print("Using Python interpreter:", sys.executable)


import torch
import torch.nn as nn
import numpy as np
import chess
import os
import json
import re
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

directories = ('../data/processed_games_4', )
move_dict = '../data/moves.json'
game_data = []

with open(move_dict, 'r') as file:
    move_dict = json.load(file)

for directory in directories:
    for dirpath, dirnames, filenames in os.walk(directory):
        if directory != '..data/processed_tactics':
            if dirpath == directory:
                continue

        print('Subdirectory:', dirpath)

        for filename in filenames:
            if filename.endswith('.npz'):
                file_path = os.path.join(dirpath, filename)
                data = np.load(file_path)

                if 'state' in data and 'correct_move' in data and 'fen' in data:
                    state_tensor = torch.from_numpy(data['state'])

                    correct_move = data['correct_move'] 
                    correct_move = str(correct_move)
                    cleaned = re.sub(r"[+|#|=].*?(?=\s|$)", "", correct_move)
                    move = move_dict[cleaned]

                    fen = data['fen'].item()

                    game_data.append((state_tensor, move, fen))
                else:
                    print(f"Required keys not found in {file_path}")

class ChessDataset(Dataset):
    def __init__(self, game_data):
        self.game_data = game_data

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, idx):
        state, move, fen = self.game_data[idx]
        return state, move, fen

# Assuming `game_data` is a list of tuples (state_tensor, correct_move)
dataset = ChessDataset(game_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('../models/model_epoch_15.pth'))

def are_legal_moves(moves, fens):
    penalty_t = torch.ones_like(moves, dtype=torch.float)
    illegal_cnt = 0
    for i, (move, fen) in enumerate(zip(moves, fens)):
        key = next((key for key, value in move_dict.items() if value == move), None)
        board = chess.Board(fen)
        try:
            board.push_san(key)
        except:
            #move is illegal
            illegal_cnt += 1
            penalty_t[i] = 2
    return penalty_t, illegal_cnt

def adjust_logits(logits, legal_moves_masks):
    """
    Adjusts logits for a batch of game states, penalizing illegal moves.
    
    :param logits: A 2D tensor of logits from the model (batch_size x num_moves).
    :param legal_moves_masks: A boolean tensor indicating legal moves (batch_size x num_moves).
    :return: Adjusted logits.
    """
    illegal_moves_penalty = -1e9
    inverse_mask = 1 - mask
    adjusted_logits = logits + (inverse_mask.float() * illegal_moves_penalty)   
    return adjusted_logits

num_epochs = 30
log_interval = 10 

for epoch in range(num_epochs): 
    model.train()  # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (board_states, correct_moves, fens) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(board_states)
        #outputs containing all possible moves
        #need to know what all moves are
        #for move in move dict
        #see if move is legal in specific position
        #if not, set the logit to a very small value
        """batch_size = outputs.size(0)
        num_moves = outputs.size(1)
        mask = torch.zeros(batch_size, num_moves, dtype=torch.float32)
        i = 0
        for fen in fens:
            board = chess.Board(fen)
            move_no = 0
            for move in move_dict:
                try:
                    board.push_san(move)
                    board.pop()
                    mask[i][move_no] = 1
                except:
                    mask[i][move_no] = 0
                move_no += 1
            i+=1
        new_logits = adjust_logits(outputs, mask)"""
        _, predicted_moves = torch.max(outputs, dim=1)
        # Calculate loss
        loss = criterion(outputs, correct_moves)
        total_loss += loss.item()
        
        # Perform backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        correct_predictions += (predicted_moves == correct_moves).sum().item()
        total_predictions += correct_moves.size(0)
        
        # Log training progress
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100. * correct_predictions / total_predictions:.2f}%')
    
    # Log epoch statistics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct_predictions / total_predictions
    print(f'End of Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Save model checkpoint
    torch.save(model.state_dict(), f'../models/model_epoch_{epoch+1}.pth')

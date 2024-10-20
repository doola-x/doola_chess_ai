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

directories = ('../data/processed_games_4', '../data/processed_games_3', '../data/processed_games_2', '../data/processed_games')
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
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc = nn.Linear(1024, 9010)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjusting dimensions for the Conv2D layer
        x = self.relu(self.conv1(x))  # Pass through the first convolutional layer and ReLU
        x = self.flatten(x)  # Flatten the output to feed into LSTM
        x = x.unsqueeze(1)  # Adjusting the shape for LSTM (batch_size, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1]
        x = self.fc(x)
        return x

model = ChessModel()

# Set the device to MPS if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)  # Move the model to the appropriate device

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('../models/ac/actor_epoch_45.pth', map_location=device))

def are_legal_moves(moves, fens):
    penalty_t = torch.ones_like(moves, dtype=torch.float).to(device)
    illegal_cnt = 0
    for i, (move, fen) in enumerate(zip(moves, fens)):
        key = next((key for key, value in move_dict.items() if value == move), None)
        board = chess.Board(fen)
        try:
            board.push_san(key)
        except:
            illegal_cnt += 1
            penalty_t[i] = 2
    return penalty_t, illegal_cnt

def adjust_logits(logits, legal_moves_masks):
    illegal_moves_penalty = -1e9
    inverse_mask = 1 - mask
    adjusted_logits = logits + (inverse_mask.float() * illegal_moves_penalty)
    return adjusted_logits

num_epochs = 30
log_interval = 10 

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (board_states, correct_moves, fens) in enumerate(dataloader):
        # Move data to the appropriate device
        board_states, correct_moves = board_states.to(device), correct_moves.to(device)
        
        optimizer.zero_grad()
        outputs = model(board_states)
        _, predicted_moves = torch.max(outputs, dim=1)
        
        # Calculate loss
        loss = criterion(outputs, correct_moves)
        total_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        correct_predictions += (predicted_moves == correct_moves).sum().item()
        total_predictions += correct_moves.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100. * correct_predictions / total_predictions:.2f}%')
    
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct_predictions / total_predictions
    print(f'End of Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Save model checkpoint
    torch.save(model.state_dict(), f'../models/model_epoch_{epoch+1}.pth')

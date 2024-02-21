import torch
import torch.nn as nn
import numpy as np
import chess
import os
import json
import re
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

directory = '../data/processed_games'
move_dict = '../data/moves.json'
game_data = []

with open(move_dict, 'r') as file:
    move_dict = json.load(file)


for dirpath, dirnames, filenames in os.walk(directory):
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
                print('Tensor shape:', state_tensor.shape, 'Correct move:', move, 'Fen:', fen)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def are_legal_moves(moves, fens):
    penalty_t = torch.zeros_like(moves, dtype=torch.float)
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

# Assuming you've already defined your model, dataloader, criterion, and optimizer

num_epochs = 50 # Number of epochs to train for
log_interval = 10  # Interval for printing log information

for epoch in range(num_epochs): 
    model.train()  # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (board_states, correct_moves, fens) in enumerate(dataloader):
        optimizer.zero_grad()
        #board_states_flattened = board_states.view(board_states.size(0), -1)
        outputs = model(board_states)
        _, predicted_moves = torch.max(outputs, dim=1)
        pen_flag, errs = are_legal_moves(predicted_moves, fens)
        # Calculate loss
        loss = criterion(outputs, correct_moves)
        loss = (loss * pen_flag).mean()
        total_loss += loss.item()
        
        # Perform backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        
        correct_predictions += (predicted_moves == correct_moves).sum().item()
        total_predictions += correct_moves.size(0)
        
        # Log training progress
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100. * correct_predictions / total_predictions:.2f}%, Illegal Suggestions: {errs}')
    
    # Log epoch statistics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct_predictions / total_predictions
    print(f'End of Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Save model checkpoint
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

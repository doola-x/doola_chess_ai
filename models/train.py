import torch
import torch.nn as nn
import numpy as np
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

            if 'state' in data and 'correct_move' in data:
                state_tensor = torch.from_numpy(data['state'])
                correct_move = data['correct_move'] 
                correct_move = str(correct_move)
                cleaned = re.sub(r"[+|#|=].*?(?=\s|$)", "", correct_move)
                move = move_dict[cleaned]

                game_data.append((state_tensor, move))
                print('Tensor shape:', state_tensor.shape, 'Correct move:', move)
            else:
                print(f"Required keys not found in {file_path}")

class ChessDataset(Dataset):
    def __init__(self, game_data):
        self.game_data = game_data

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, idx):
        state, move = self.game_data[idx]
        return state, move

# Assuming `game_data` is a list of tuples (state_tensor, correct_move)
dataset = ChessDataset(game_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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


model = ChessModel(input_size=832, output_size=9010)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assuming you've already defined your model, dataloader, criterion, and optimizer

num_epochs = 50 # Number of epochs to train for
log_interval = 10  # Interval for printing log information

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (board_states, correct_moves) in enumerate(dataloader):
        optimizer.zero_grad()
        board_states_flattened = board_states.view(board_states.size(0), -1)
        outputs = model(board_states_flattened.float())
        
        # Calculate loss
        loss = criterion(outputs, correct_moves)
        total_loss += loss.item()
        
        # Perform backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted_moves = torch.max(outputs, dim=1)
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
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

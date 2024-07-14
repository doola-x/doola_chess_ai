import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)  # 64 filters, kernel size 3x3
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.reshape(-1, 64 * 8 * 8)  # Flatten the output of the conv layer
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class ChessDataset(Dataset):
    def __init__(self, game_data):
        self.game_data = game_data

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, idx):
        state, move = self.game_data[idx]
        return state, move

directories = ('../data/processed_value',)
game_data = []

for directory in directories:
    for dirpath, dirnames, filenames in os.walk(directory):

        print('Subdirectory:', dirpath)

        for filename in filenames:
            if filename.endswith('.npz'):
                file_path = os.path.join(dirpath, filename)
                data = np.load(file_path)

                if 'state' in data and 'correct_move' in data:
                    state_tensor = torch.from_numpy(data['state'])
                    correct_move = data['correct_move'] 
                    new_move = float(correct_move)

                    game_data.append((state_tensor, new_move))
                    print(f"state: {state_tensor}, correct val: {new_move} of type: {type(new_move)}")
                else:
                    print(f"Required keys not found in {file_path}")

dataset = ChessDataset(game_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ChessValueNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

num_epochs = 30
log_interval = 10 

for epoch in range(num_epochs): 
    model.train()  # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (board_states, correct_moves) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(board_states)
        #print(f"outputs: {outputs}")
        outputs = outputs.reshape(-1)
        # Calculate loss
        #print(f"correct moves as float: {correct_moves.type_as(outputs)}")
        loss = criterion(outputs.type_as(correct_moves), correct_moves)
        total_loss += loss.item()
        
        # Perform backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        correct_predictions += (outputs == correct_moves).sum().item()
        total_predictions += correct_moves.size(0)
        
        # Log training progress
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100. * correct_predictions / total_predictions:.2f}%')
    
    # Log epoch statistics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct_predictions / total_predictions
    print(f'End of Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Save model checkpoint
    torch.save(model.state_dict(), f'../models/value/model_epoch_{epoch+1}.pth')
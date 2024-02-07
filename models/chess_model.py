import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

directory = '../data/processed_games/'

all_games = []  # List to store all game data

for filename in os.listdir(directory):
    if filename.endswith('.npz'):
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)
        games = data['states']  # Replace 'games' with the correct key if different
        all_games.extend(games)

print(all_games)

# Assuming X_train and y_train are your data and labels
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class ChessModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(ChessModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Get the output of the last sequence
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = ChessModel(input_size, hidden_size=64, output_size=num_possible_moves)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
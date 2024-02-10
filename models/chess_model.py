import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

directory = '../data/processed_games'
sequences = []

for dirpath, dirnames, filenames in os.walk(directory):
    if dirpath == directory:
        continue

    print('Subdirectory:', dirpath)
    sequence = []
    #pad all 0's tensor to signal game start
    sequence.append(torch.zeros((8, 8, 13)))
    for filename in filenames:
        if filename.endswith('.npz'):
            file_path = os.path.join(dirpath, filename)
            data = np.load(file_path)
            
            if 'states' in data:
                tensor = torch.from_numpy(data['states'])
                sequence.append(tensor)
                print('Tensor shape:', tensor.shape)
            else:
                print(f"Key 'states' not found in {file_path}")
    #arbitrary 1's padding to signal end of game
    while (len(sequence) < 100):
        sequence.append(torch.ones((8, 8, 13)))
    sequences.append(sequence)


train_loader = DataLoader(dataset=sequences, batch_size=32, shuffle=True)

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
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Assuming the output of conv layers is flattened, adjust the input size accordingly
        # LSTM input dimensions: (batch_size, seq_len, features)
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc = nn.Linear(1024, 9010)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))  # Pass through the first convolutional layer and ReLU
        x = self.flatten(x)  # Flatten the output to feed into LSTM
        x = x.unsqueeze(1)  # Adjusting the shape for LSTM (batch_size, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Using the last layer's hidden state. You mentioned using lstm_out, but usually hn[-1] is used
        # Using lstm_out's last sequence output if the LSTM's `batch_first=True`
        x = hn[-1]  # If you're sure about using lstm_out, replace `hn[-1]` with `lstm_out[:, -1, :]`
        x = self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
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


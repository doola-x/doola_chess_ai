import torch
import torch.nn as nn

class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # Dense layers
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 1)  # Output scalar value

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # Scalar value output
        return x

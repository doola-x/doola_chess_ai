import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(64)
		self.fc1 = nn.Linear(64 * 8 * 8, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024, 512)
		self.bn3 = nn.BatchNorm1d(512)
		self.fc3 = nn.Linear(512, 4208)

	def forward(self, x):
		x = x.permute(0, 3, 1, 2)
		x = F.relu(self.bn1(self.conv1(x)))
		x = x.reshape(-1, 64 * 8 * 8)  # Flatten the output of the conv layer
		x = F.relu(self.bn2(self.fc1(x)))
		x = F.relu(self.bn3(self.fc2(x)))
		x = torch.sigmoid(self.fc3(x))
		return x
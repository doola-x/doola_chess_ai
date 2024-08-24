import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

def fen_to_tensor(fen):
	board_tensor = np.zeros((8, 8, 13), dtype=np.float32)
	
	pieces, active_color, _, _, _, _ = fen.split(' ')
	rows = pieces.split('/')

	for i, row in enumerate(rows):
		col = 0
		for char in row:
			if char.isdigit():
				col += int(char)
			else:
				piece_idx = piece_to_idx[char]
				board_tensor[i, col, piece_idx] = 1
				col += 1
				
	if active_color == 'w':
		board_tensor[:, :, 12] = 1  # Indicate active color is white
	else:
		board_tensor[:, :, 12] = 0  # Indicate active color is black
	
	return torch.tensor(board_tensor)


def main(fen): 
	model = ChessValueNetwork()
	model.load_state_dict(torch.load('../models/value/model_epoch_15.pth'))
	model.eval()
	tensor = fen_to_tensor(fen)
	tensor = tensor.unsqueeze(0)
	with torch.no_grad():
		prediction = model(tensor)
		print(f"{prediction}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Predict scalar value of a position from a given board state.")
	parser.add_argument("fen", help="FEN string representing the board state")
	args = parser.parse_args()
	
	main(args.fen)
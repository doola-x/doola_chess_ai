import numpy as np

data = np.load('game_data.npz', allow_pickle=True)
game_states = data['states']
game_color = data['color']
print(game_color)
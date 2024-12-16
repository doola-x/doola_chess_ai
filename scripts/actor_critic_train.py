import json
import torch
import chess
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from models import Critic
from rl_arch import Actor
from chess_env import ChessEnvironment 

# Initialize Actor
actor = Actor()
actor.train()
actor.load_state_dict(torch.load('../models/actor0_epoch_9.pth'))

# Initialize Optimizer
actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)

# Initialize Environment
env = ChessEnvironment()
env.start_engine()

# Piece mapping
piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    ' ': 12  # Empty square
}

# FEN to tensor conversion
def fen_to_tensor(fen):
    board_tensor = torch.zeros(8, 8, 13)
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
    return board_tensor

# Decode move from index to SAN notation
def decode_move(value, filename='../data/moves0.json'):
    try:
        with open(filename, 'r') as file:
            mapping = json.load(file)
            for key, val in mapping.items():
                if val == value:
                    return key
            raise ValueError(f"No move found for value {value}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found.")

# Training loop
def train_actor(episodes):
    best_total_reward = float('-inf')
    for episode in range(episodes):
        state = env.reset_board()
        total_reward = 0
        total_loss = 0
        moves = 1
        done = False

        while not done:
            tensor = fen_to_tensor(env.board.fen()).unsqueeze(0)  # [1, 13, 8, 8]
            print(f"FEN Tensor Shape: {tensor.shape}")

            probabilities = actor(tensor)
            probabilities = F.softmax(torch.flatten(probabilities), dim=0)  # Normalize probabilities
            top_k = torch.topk(probabilities, 5)

            reward_paths = []
            for i in range(len(top_k[0])):
                action = decode_move(top_k[1][i].item())
                if env.is_legal_move(action):
                    potential_reward, done = env.step(action, top_k[1][i], True)
                    log_prob = torch.log(top_k[0][i])
                    reward_paths.append((potential_reward, log_prob, action))

            if len(reward_paths) == 0:
                legal_action = env.get_legal_move()
                print(f"No valid reward paths. Using legal action: {legal_action}")
                reward, done = env.step(legal_action, moves, False)
            else:
                top_reward_paths = sorted(reward_paths, key=lambda x: x[0], reverse=True)
                reward, done = env.step(top_reward_paths[0][2], moves, False)
                log_prob = top_reward_paths[0][1]

            # Update rewards and losses
            total_reward += reward
            actor_loss = -log_prob * total_reward / moves
            total_loss += actor_loss.item()

            # Backpropagate actor loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            print(f"Move {moves}: Action {top_reward_paths[0][2]}, Log Prob: {log_prob.item()}, Reward: {reward}, Total Reward: {total_reward}")
            moves += 1

        # Game outcome scaling
        outcome = env.board.outcome()
        if outcome.winner == chess.WHITE:
            total_reward *= 10
        elif outcome.winner is None:
            total_reward = total_reward  # No change
        else:
            total_reward *= 2

        # Save model if reward improves
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            torch.save(actor.state_dict(), f'../models/ac/actor_best.pth')

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Moves: {moves}, Winner: {outcome.winner}")

    print("Training complete.")

# Start training
train_actor(1000)
env.stop_engine()

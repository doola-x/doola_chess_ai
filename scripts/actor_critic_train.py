import json
import torch
import chess
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from models import Actor, Critic
from chess_env import ChessEnvironment 

actor = Actor()
#critic = Critic()
actor.train()
#critic.train()
actor.load_state_dict(torch.load('../models/model_epoch_3.pth'))
#critic.load_state_dict(torch.load('../models/value/model_epoch_30.pth'))

actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
#critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

env = ChessEnvironment()
env.start_engine()

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
    
    return torch.tensor(board_tensor).unsqueeze(0)

def decode_move(value, filename='../data/moves.json'):
    """Decodes a move from the model's output integer back to SAN notation by loading a mapping from a file."""
    try:
        with open(filename, 'r') as file:
            mapping = json.load(file)
            # Reverse the mapping to find the key by value
            for key, val in mapping.items():
                if val == value:
                    return key
            # If no matching value is found
            raise ValueError(f"No move found for value {value}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found.")

def train_actor_critic(episodes):
    best_total_reward = float('inf')
    for episode in range(episodes):
        state = env.reset_board()  # Start a new game
        #print(state)
        total_reward = 0
        total_loss = 0
        done = False
        moves = 1
        while not done:
            tensor = fen_to_tensor(env.board.fen())
            #print("Shape after fen_to_tensor:", tensor.shape)
            #tensor = tensor.repeat(2, 1, 1, 1)
            probabilities = actor(tensor)
            probabilities = torch.flatten(probabilities)
            #print(f"probabilities: {probabilities}, probabilities shape: {probabilities.shape}")
            # Sort the probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            moves_tosearch = []
            probs_tosearch = []
            moves_c = 0
            print(f"len: {len(sorted_probs)}")
            for i in range(len(sorted_probs)):
                #print(f"Probability shape: {sorted_probs.shape}, Index: {sorted_indices[i]}")
                action = decode_move(sorted_indices[i])
                print(f"action: {action}, i: {i}")
                legal = env.is_legal_move(action)
                if (legal):
                    moves_tosearch.append(action)
                    probs_tosearch.append(sorted_probs[i])
                    if (len(moves_tosearch) > 30):
                        break
                    #moves_c += 1
                    #if (moves_c == 5):
                    #    break
                    """predicted_move = action
                    probability = sorted_probs[i]
                    prob_idx = i
                    break"""
            #probability, predicted_move = torch.max(probabilities, dim=1)
            #print(probability)
            move = random.choices(moves_tosearch, probs_tosearch, k=1)[0]
            reward, done = env.step(moves_tosearch[n], moves)
            #print(reward)
            """next_tensor = fen_to_tensor(env.board.fen())
            tensor_dup = tensor.repeat(2, 1, 1, 1)
            next_tensor_dup = next_tensor.repeat(2, 1, 1, 1)
            current_value = critic(tensor_dup)
            print(f"current value prediction: {current_value}")
            next_value = critic(next_tensor_dup)
            print(f"next value prediction: {next_value}")"""
            
            # calculate the target value using reward and the value prediction of the next state
            """target_value = reward + 0.99 * next_value
            #print(target_value)
            # Critic's loss: Mean Squared Error between predicted value and target value
            critic_loss = F.mse_loss(current_value, target_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()"""
            
            # Actor's loss: typically -log(probability of the action) * advantage
            # Advantage = target value - current value prediction
            #advantage = (target_value - current_value).detach()  # Stop gradient flow here
            #print("Advantage shape:", advantage.shape)
            #epsilon = 1e-8
            log_prob = torch.log(probs_tosearch[moves_tosearch.index(move)])
            actor_loss = (log_prob * reward) * (1/moves*2)
            total_loss += actor_loss
            #if actor_loss.dim() > 0:
            #    actor_loss = actor_loss.mean()
            #print("Actor loss shape after reduction:", actor_loss.shape)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            moves += 1
            total_reward += reward
            last_fen = env.board.fen()
            print(f"predicted move: {move}, prob log: {log_prob}, reward: {reward}, total actor loss: {total_loss}, total reward: {total_reward}\n", end="")
        outcome = env.board.outcome()
        if (outcome.winner == chess.WHITE):
            if (reward < 0):
                reward *= -1
            else:
                reward *= 10
        elif (outcome.winner == None):
            #hm
            reward = reward
        else:
            reward *= 2
        if episode % 10 == 0 or total_reward > best_total_reward:
            best_total_reward = total_reward
            torch.save(actor.state_dict(), f'../models/ac/actor_epoch_{episode+1}.pth')
            #torch.save(critic.state_dict(), f'../models/ac/critic_epoch_{episode+1}.pth')
        # Logging and monitoring
        print(f'\n======================================================================\nepisode {episode}: total reward = {total_reward}, last fen:{last_fen}, winner: {outcome.winner}, total moves: {moves}\n======================================================================\n')

# Number of episodes to train
train_actor_critic(1000)
env.stop_engine()

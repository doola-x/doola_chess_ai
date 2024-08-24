import json
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from models import Actor, Critic
from chess_env import ChessEnvironment 

actor = Actor()
critic = Critic()
actor.train()
critic.train()
actor.load_state_dict(torch.load('../models/model_epoch_30.pth'))
critic.load_state_dict(torch.load('../models/value/model_epoch_30.pth'))

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

env = ChessEnvironment()

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
        done = False
        env.push_legal_move()
        moves = 0
        while not done:
            #print(env.board.fen())
            tensor = fen_to_tensor(env.board.fen())
            #print("Shape after fen_to_tensor:", tensor.shape)
            #tensor = tensor.repeat(2, 1, 1, 1)
            probabilities = actor(tensor)
            probability, predicted_move = torch.max(probabilities, dim=1)
            #print(probability)
            action = decode_move(predicted_move.item())
            #print(action)

            reward, done = env.step(action)
            #print(reward)
            next_tensor = fen_to_tensor(env.board.fen())
            tensor_dup = tensor.repeat(2, 1, 1, 1)
            next_tensor_dup = next_tensor.repeat(2, 1, 1, 1)
            current_value = critic(tensor_dup)
            next_value = critic(next_tensor_dup)
            
            # calculate the target value using reward and the value prediction of the next state
            target_value = reward + 0.99 * next_value
            #print(target_value)
            # Critic's loss: Mean Squared Error between predicted value and target value
            critic_loss = F.mse_loss(current_value, target_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Actor's loss: typically -log(probability of the action) * advantage
            # Advantage = target value - current value prediction
            advantage = (target_value - current_value).detach()  # Stop gradient flow here
            #print("Advantage shape:", advantage.shape)
            epsilon = 1e-8
            actor_loss = -torch.log(probability + epsilon) * advantage
            #print("Actor loss shape before reduction:", actor_loss.shape)
            if actor_loss.dim() > 0:
                actor_loss = actor_loss.mean()
            #print("Actor loss shape after reduction:", actor_loss.shape)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            moves += 1
            if (done == False): 
                env.push_legal_move()
                moves += 1
            total_reward += reward
            print(f"total reward: {total_reward}")
            if episode % 100 == 0 or total_reward < best_total_reward:
                best_total_reward = total_reward
                torch.save(actor.state_dict(), f'../models/ac/actor_epoch_{episode+1}.pth')
                torch.save(critic.state_dict(), f'../models/ac/critic_epoch_{episode+1}.pth')
            if done:
                break
        
        # Logging and monitoring
        print(f'Episode {episode}: Total Reward = {total_reward}')

# Number of episodes to train
train_actor_critic(1000)

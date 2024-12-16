import os
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from rl_arch import Actor
from torch.utils.tensorboard import SummaryWriter

game_data = []

def load_compressed(filename):
    with gzip.open(filename, 'rb') as f:
        data = torch.load(f)
        return data['state'], data['action']

def train_actor_model(data_dir, num_epochs=30, batch_size=32, learning_rate=0.01, save_dir="../models"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.gz'):
                state, action = load_compressed(os.path.join(dirpath, filename))
                output_t = torch.zeros(4208)
                output_t[action] = 1
                game_data.append((state, output_t))
    print(f"Loaded {len(game_data)} data samples.")
    data_loader = torch.utils.data.DataLoader(game_data, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    actor = Actor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    actor.train()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}...")
        for batch_idx, (state, action) in enumerate(data_loader):
            outputs = actor(state)
            loss = criterion(outputs, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log batch loss
            if batch_idx % 10 == 0:
                print(f"\r  Batch {batch_idx}/{len(data_loader)} - Loss: {loss.item():.4f}")

        # Log epoch loss
        print(f"Epoch {epoch+1} - Total Loss: {total_loss:.4f}")

        # Save model weights at the end of the epoch
        model_save_path = os.path.join(save_dir, f"actor0_epoch_{epoch+1}.pth")
        torch.save(actor.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

    print("Training complete.")
    return actor

# Example usage
model = train_actor_model('../data/processed_0')
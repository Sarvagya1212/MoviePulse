# train_autoencoder.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle

print("--- Step 1: Preparing User-Item Matrix ---")

# Load the data
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

# Create the user-item matrix
n_users = ratings['user_id'].nunique()
n_items = ratings['movie_id'].nunique()

# Use pivot_table to create the matrix, fill missing values with 0
user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Get the mapping for movie_ids to matrix columns
movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(user_item_matrix.columns)}
idx_to_movie_id = {i: movie_id for movie_id, i in movie_id_to_idx.items()}

print(f"Created a user-item matrix of shape: {user_item_matrix.shape}") # (943 users, 1682 movies)


class AutoencoderCF(nn.Module):
    def __init__(self, n_items, hidden_dim=128, latent_dim=32):
        super(AutoencoderCF, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items),
            # Using Sigmoid to scale output between 0 and 1, we will scale it up later
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Custom Dataset for our matrix
class UserRatingsDataset(Dataset):
    def __init__(self, matrix):
        self.matrix = torch.FloatTensor(matrix.values)

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return self.matrix[idx]
    
    
    
    print("\n--- Step 2: Model Training ---")

# --- Training Setup ---
dataset = UserRatingsDataset(user_item_matrix)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# n_items should be the number of columns in our matrix
model = AutoencoderCF(n_items=user_item_matrix.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
n_epochs = 25 # Autoencoders often need more epochs to converge
for epoch in range(n_epochs):
    total_loss = 0
    for data in dataloader:
        # Get the input batch
        inputs = data
        
        # Get model outputs
        reconstructed = model(inputs)
        
        # Create a mask to calculate loss only on non-zero ratings
        mask = inputs > 0
        loss = criterion(reconstructed[mask], inputs[mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

print("\nTraining complete.")

# --- Save the Model and Mappings ---
print("\n--- Step 3: Saving Model and Mappings ---")
model_data = {
    'model_state_dict': model.state_dict(),
    'n_items': user_item_matrix.shape[1],
    'movie_id_to_idx': movie_id_to_idx,
    'idx_to_movie_id': idx_to_movie_id
}

with open('autoencoder_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Autoencoder model and data saved to 'autoencoder_model.pkl'")




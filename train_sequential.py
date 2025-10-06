# train_sequential.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle


# Load the data
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

# We need to map movie_ids to a continuous range of integers for the embedding layer
unique_movie_ids = ratings['movie_id'].unique()
movie_id_map = {id: i for i, id in enumerate(unique_movie_ids)}
n_items = len(unique_movie_ids)

# Sort interactions by user and time
ratings = ratings.sort_values(by=['user_id', 'timestamp'])

# Group ratings by user to create sequences
sequences = ratings.groupby('user_id')['movie_id'].apply(list)

# Create input sequences and target labels for the RNN
# Example: For a sequence [A, B, C, D], the training pairs are:
# ([A], B), ([A, B], C), ([A, B, C], D)
X = []
y = []
for seq in sequences:
    for i in range(1, len(seq)):
        # Map original movie_ids to our new continuous indices
        input_seq = [movie_id_map[movie_id] for movie_id in seq[:i]]
        target_item = movie_id_map[seq[i]]
        X.append(input_seq)
        y.append(target_item)

print(f"Generated {len(X)} input/target sequence pairs.")



class GRUModel(nn.Module):
    def __init__(self, n_items, embedding_dim=32, hidden_dim=64, n_layers=1):
        super(GRUModel, self).__init__()
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items)

    def forward(self, x, h=None):
        # x is a padded batch of sequences
        x = self.embedding(x)
        out, h = self.gru(x, h)
        # We only want the output from the last time step
        out = self.fc(out[:, -1, :])
        return out, h
    

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def collate_fn(batch):
    # Separate sequences and targets
    sequences, targets = zip(*batch)
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return sequences_padded, targets

# --- Training Setup ---
print("\n--- Step 2: Model Training ---")
dataset = SequenceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

model = GRUModel(n_items=n_items)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
n_epochs = 5 # For a real model, you might use 10-20 epochs
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for sequences_padded, targets in dataloader:
        optimizer.zero_grad()
        # The model returns outputs and the final hidden state, we only need outputs
        outputs, _ = model(sequences_padded)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

print("\nTraining complete.")

# --- Save the Model and Mappings ---
print("\n--- Step 3: Saving Model and Mappings ---")
model_data = {
    'model_state_dict': model.state_dict(),
    'n_items': n_items,
    'movie_id_map': movie_id_map,
    'unique_movie_ids': unique_movie_ids
}

with open('sequential_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Sequential model and data saved to 'sequential_model.pkl'")
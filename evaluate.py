# evaluate.py
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

# --- Helper Functions for Metrics ---
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    return precisions, recalls

def ndcg_at_k(predictions, k=10):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        true_relevance = np.asarray([true_r for (_, true_r) in user_ratings])
        est_relevance = np.asarray([est for (est, _) in user_ratings])

        n = min(len(true_relevance), k)  # Ensure no broadcast error
        if n == 0:
            ndcgs[uid] = 0
            continue

        dcg = np.sum(true_relevance[:n] / np.log2(np.arange(2, n + 2)))
        ideal_relevance = np.sort(true_relevance)[::-1]
        idcg = np.sum(ideal_relevance[:n] / np.log2(np.arange(2, n + 2)))

        ndcgs[uid] = dcg / idcg if idcg > 0 else 0

    return ndcgs


# --- 1. Load Data and Split ---
print("--- Loading and Splitting Data ---")
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = surprise_split(data, test_size=0.2)

# --- 2. Evaluate SVD Model ---
print("\n--- Evaluating SVD Model ---")
svd_model = SVD()
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)

svd_precisions, svd_recalls = precision_recall_at_k(svd_predictions, k=10)
svd_ndcgs = ndcg_at_k(svd_predictions, k=10)

avg_svd_precision = sum(p for p in svd_precisions.values()) / len(svd_precisions)
avg_svd_recall = sum(r for r in svd_recalls.values()) / len(svd_recalls)
avg_svd_ndcg = sum(n for n in svd_ndcgs.values()) / len(svd_ndcgs)

# --- 3. Evaluate Autoencoder Model ---
print("\n--- Evaluating Autoencoder Model ---")
# Re-create user-item matrix for training set ONLY
train_df = pd.DataFrame(trainset.all_ratings(), columns=['user_id', 'movie_id', 'rating'])
user_item_matrix_train = train_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
all_movie_ids = ratings['movie_id'].unique()
user_item_matrix_train = user_item_matrix_train.reindex(columns=all_movie_ids, fill_value=0)

# Autoencoder classes (must be defined here as well)
class AutoencoderCF(nn.Module):
    def __init__(self, n_items, hidden_dim=128, latent_dim=32):
        super(AutoencoderCF, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(n_items, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_items), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class UserRatingsDataset(TorchDataset):
    def __init__(self, matrix): self.matrix = torch.FloatTensor(matrix.values)
    def __len__(self): return len(self.matrix)
    def __getitem__(self, idx): return self.matrix[idx]

# Retrain Autoencoder on the training set
n_items_ae = user_item_matrix_train.shape[1]
ae_dataset = UserRatingsDataset(user_item_matrix_train)
ae_dataloader = DataLoader(ae_dataset, batch_size=32, shuffle=True)
ae_model = AutoencoderCF(n_items=n_items_ae)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

for epoch in range(10): # Shorter training for evaluation
    for data in ae_dataloader:
        reconstructed = ae_model(data)
        mask = data > 0
        loss = criterion(reconstructed[mask], data[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Generate predictions for the test set
ae_model.eval()
ae_predictions = []
with torch.no_grad():
    full_reconstructed = ae_model(torch.FloatTensor(user_item_matrix_train.values)) * 4 + 1
for uid, iid, true_r in testset:
    try:
        user_idx = trainset.to_inner_uid(uid)
        item_idx = trainset.to_inner_iid(iid)
        est = full_reconstructed[user_idx, item_idx].item()
        ae_predictions.append((uid, iid, true_r, est, {}))
    except (ValueError, IndexError):
        continue

ae_precisions, ae_recalls = precision_recall_at_k(ae_predictions, k=10)
ae_ndcgs = ndcg_at_k(ae_predictions, k=10)

avg_ae_precision = sum(p for p in ae_precisions.values()) / len(ae_precisions)
avg_ae_recall = sum(r for r in ae_recalls.values()) / len(ae_recalls)
avg_ae_ndcg = sum(n for n in ae_ndcgs.values()) / len(ae_ndcgs)

# --- 4. Display Leaderboard ---
print("\n--- MODEL EVALUATION LEADERBOARD (Top-10 Recommendations) ---")
print("-" * 60)
print(f"| {'Model':<15} | {'Precision@10':<15} | {'Recall@10':<12} | {'NDCG@10':<10} |")
print("-" * 60)
print(f"| {'SVD':<15} | {avg_svd_precision:<15.4f} | {avg_svd_recall:<12.4f} | {avg_svd_ndcg:<10.4f} |")
print(f"| {'Autoencoder':<15} | {avg_ae_precision:<15.4f} | {avg_ae_recall:<12.4f} | {avg_ae_ndcg:<10.4f} |")
print("-" * 60)
# train.py
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

print("--- Starting Model Training ---")

# Load the data using Surprise's Reader
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# --- 2.1.1 Memory-Based Method: Item-Based Collaborative Filtering ---
print("\nTraining Item-Based Collaborative Filtering (KNN)...")
# Using cosine similarity
sim_options = {'name': 'cosine', 'user_based': False}
knn_model = KNNWithMeans(sim_options=sim_options)
knn_model.fit(trainset)

# --- 2.1.2 Model-Based Method: Matrix Factorization (SVD) ---
print("\nTraining Matrix Factorization (SVD)...")
svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd_model.fit(trainset)

# --- 4.1 Evaluate Models ---
print("\n--- Evaluating Models ---")
# Evaluate KNN
knn_predictions = knn_model.test(testset)
knn_rmse = accuracy.rmse(knn_predictions)
print(f"Item-Based CF RMSE: {knn_rmse}")

# Evaluate SVD
svd_predictions = svd_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions)
print(f"SVD RMSE: {svd_rmse}")

# The model with the lower RMSE is generally better. Let's save the SVD model.
print("\nSaving the SVD model to 'svd_model.pkl'...")
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)
    
print("\nModel training and saving complete!")
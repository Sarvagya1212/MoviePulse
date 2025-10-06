import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
import pickle

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)


# Total combinations: 2 (epochs) * 2 (lr) * 2 (reg) * 2 (factors) = 16
param_grid = {
    'n_epochs': [10, 20],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1],
    'n_factors': [50, 100]
}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures=['rmse', 'mae'],
    cv=3,
    n_jobs=-1, # Use all available CPU cores
    refit=True
)

print("Running Grid Search... (This may take several minutes)")
gs.fit(data)

print("\n Grid Search Complete")

print(f"Best RMSE score: {gs.best_score['rmse']:.4f}")

print("Best parameters:")
print(gs.best_params['rmse'])


best_model = gs.best_estimator['rmse']

with open('svd_tuned_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nBest SVD model has been tuned and saved to 'svd_tuned_model.pkl'")
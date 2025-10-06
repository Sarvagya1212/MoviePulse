# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# u.data format: user_id, item_id, rating, timestamp
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

# u.item format: movie_id | movie_title | ...
movies_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, usecols=range(2), encoding='latin-1')

# Merge the dataframes
data = pd.merge(ratings, movies, on='movie_id')

print("## Data Head ##")
print(data.head())

# --- 1.1 Data Understanding & Quality Assessment ---
n_users = data['user_id'].nunique()
n_movies = data['movie_id'].nunique()
n_ratings = len(data)
sparsity = 1.0 - (n_ratings / (n_users * n_movies))

print(f"\nNumber of users: {n_users}")
print(f"Number of movies: {n_movies}")
print(f"Number of ratings: {n_ratings}")
print(f"Sparsity of the dataset: {sparsity:.2%}")

# --- Rating Distribution Analysis ---
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=data)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('rating_distribution.png')
print("\nGenerated 'rating_distribution.png'")

# --- User/Item Popularity (Long-Tail Analysis) ---
movie_popularity = data['title'].value_counts()
print("\n## Top 10 Most Rated Movies ##")
print(movie_popularity.head(10))

user_activity = data['user_id'].value_counts()
print("\n## Top 10 Most Active Users ##")
print(user_activity.head(10))
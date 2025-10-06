# build_content_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("--- Starting Content-Based Model Build ---")

# --- Load and Prepare Data ---
# Define column names for u.item
i_cols = [
    'movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 
    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# Create a 'genres' column by combining all genre names for each movie
genre_cols = i_cols[6:] # Get genre column names
# For each movie (row), get the names of columns where the value is 1
movies['genres'] = movies[genre_cols].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)

print("Sample of movies with combined genre string:")
print(movies[['movie_id', 'title', 'genres']].head())


# build_content_model.py (continued)

# --- Build TF-IDF Model ---
# TfidfVectorizer will convert our genre strings into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the data, creating the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

print("\nShape of the TF-IDF matrix:", tfidf_matrix.shape) # (1682 movies, 20 genres)

# --- Calculate Cosine Similarity ---
# This creates a square matrix where each entry (i, j) is the similarity between movie i and movie j
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Shape of the Cosine Similarity matrix:", cosine_sim.shape) # (1682, 1682)

# --- Save the Model and Data ---
# We save the similarity matrix and the movie titles for our API
content_data = {
    'cosine_sim': cosine_sim,
    'movies_df': movies[['movie_id', 'title']]
}

with open('content_model.pkl', 'wb') as f:
    pickle.dump(content_data, f)

print("\nContent-based model and data saved to 'content_model.pkl'")
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import requests
import re
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. Initialize FastAPI App with CORS ---
app = FastAPI(title="MoviePulse Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 2. TMDb API Configuration & Cache ---
TMDB_API_KEY = "e52ee85b9fa9966a3e9db5aa141ef9cc"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/w1280"
metadata_cache = {}
session = requests.Session()

# --- 3. Define Model Classes ---
class GRUModel(nn.Module):
    def __init__(self, n_items, embedding_dim=32, hidden_dim=64, n_layers=1):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items)
    def forward(self, x, h=None):
        x, h = self.gru(self.embedding(x), h)
        return self.fc(x[:, -1, :]), h

class AutoencoderCF(nn.Module):
    def __init__(self, n_items, hidden_dim=128, latent_dim=32):
        super(AutoencoderCF, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(n_items, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_items), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))

class Feedback(BaseModel):
    user_id: int
    movie_id: int
    feedback_type: str

# --- 4. Load All Models and Data ---
print("Loading all models and data...")
with open('svd_tuned_model.pkl', 'rb') as f: svd_model = pickle.load(f)
with open('content_model.pkl', 'rb') as f:
    content_data = pickle.load(f)
    cosine_sim, movies_df = content_data['cosine_sim'], content_data['movies_df']

sequential_model_data = pickle.load(open('sequential_model.pkl', 'rb'))
n_items_seq, movie_id_map_seq = sequential_model_data['n_items'], sequential_model_data['movie_id_map']
idx_to_movie_id_seq = {v: k for k, v in movie_id_map_seq.items()}
sequential_model = GRUModel(n_items=n_items_seq)
sequential_model.load_state_dict(sequential_model_data['model_state_dict'])
sequential_model.eval()

autoencoder_data = pickle.load(open('autoencoder_model.pkl', 'rb'))
n_items_ae, movie_id_to_idx_ae = autoencoder_data['n_items'], autoencoder_data['movie_id_to_idx']
idx_to_movie_id_ae = autoencoder_data['idx_to_movie_id']
autoencoder_model = AutoencoderCF(n_items=n_items_ae)
autoencoder_model.load_state_dict(autoencoder_data['model_state_dict'])
autoencoder_model.eval()

i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
all_movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
genre_cols = i_cols[6:]
all_movies['genres_str'] = all_movies[genre_cols].apply(lambda row: ' | '.join(row.index[row == 1]), axis=1)

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
sequences = ratings.sort_values(by=['user_id', 'timestamp']).groupby('user_id')['movie_id'].apply(list)
movie_id_to_title = movies_df.set_index('movie_id').to_dict()['title']
movie_id_to_genres = all_movies.set_index('movie_id')['genres_str'].to_dict()
print("All models and data loaded successfully.")


# --- 5. Helper Functions ---
def get_tmdb_metadata(movie_title):
    if movie_title in metadata_cache: return metadata_cache[movie_title]
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return {"poster_url": None, "backdrop_url": None, "overview": "Add TMDb key to fetch details."}
    
    match = re.match(r'^(.*) \((\d{4})\)$', movie_title)
    if not match: return {"poster_url": None, "backdrop_url": None, "overview": "N/A"}
    title, year = match.groups()
    
    try:
        time.sleep(0.05)
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title, "year": year}
        response = session.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            movie_data = data['results'][0]
            poster_path = movie_data.get('poster_path')
            backdrop_path = movie_data.get('backdrop_path')
            result = {
                "poster_url": f"{TMDB_POSTER_BASE_URL}{poster_path}" if poster_path else None,
                "backdrop_url": f"{TMDB_BACKDROP_BASE_URL}{backdrop_path}" if backdrop_path else None,
                "overview": movie_data.get('overview')
            }
            metadata_cache[movie_title] = result
            return result
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for {movie_title}: {e}")
    
    metadata_cache[movie_title] = {"poster_url": None, "backdrop_url": None, "overview": "Details not found."}
    return metadata_cache[movie_title]

def add_metadata_to_recs(recs):
    for rec in recs:
        rec["genres"] = movie_id_to_genres.get(rec["movie_id"], "")
        metadata = get_tmdb_metadata(rec.get("title", ""))
        rec.update(metadata)
    return recs

def get_top_n_recommendations(user_id, n=10):
    all_movie_ids = ratings['movie_id'].unique()
    rated_movie_ids = ratings[ratings['user_id'] == user_id]['movie_id']
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids.values]
    preds = [svd_model.predict(user_id, mid) for mid in unrated_movie_ids]
    preds.sort(key=lambda x: x.est, reverse=True)
    recs = [{"movie_id": int(p.iid), "title": movie_id_to_title.get(p.iid, ""), "estimated_rating": float(p.est)} for p in preds[:n]]
    return recs

def get_content_explanation(user_id, recommended_movie_id):
    user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
    if user_ratings.empty: return "Popular in your region"
    try: rec_movie_idx = movies_df[movies_df['movie_id'] == recommended_movie_id].index[0]
    except IndexError: return "A top pick for you"
    most_similar_movie_title, max_similarity = "", -1
    for _, row in user_ratings.iterrows():
        try:
            rated_movie_idx = movies_df[movies_df['movie_id'] == row['movie_id']].index[0]
            similarity = cosine_sim[rec_movie_idx, rated_movie_idx]
            if similarity > max_similarity:
                max_similarity, most_similar_movie_title = similarity, movie_id_to_title.get(row['movie_id'])
        except IndexError: continue
    return f"Because you liked {most_similar_movie_title}" if most_similar_movie_title else "A top pick for you"

def get_hybrid_recommendations(user_id, n=10, alpha=0.7):
    user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
    user_top_movie_indices = movies_df[movies_df['movie_id'].isin(user_ratings['movie_id'])].index.tolist()
    if not user_top_movie_indices:
        recs = get_top_n_recommendations(user_id, n)
        for rec in recs: rec['explanation'] = "Popular in your region"
        return add_metadata_to_recs(recs)
    
    svd_candidates_raw = get_top_n_recommendations(user_id, n=50)
    candidate_ids = [c['movie_id'] for c in svd_candidates_raw]
    candidate_svd_scores = {c['movie_id']: c['estimated_rating'] for c in svd_candidates_raw}
    
    hybrid_scores = []
    for movie_id in candidate_ids:
        try:
            candidate_idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
            content_scores = cosine_sim[candidate_idx][user_top_movie_indices]
            avg_content_score = np.mean(content_scores) if content_scores.size > 0 else 0
            normalized_svd = (candidate_svd_scores[movie_id] - 1) / 4
            hybrid_score = (alpha * normalized_svd) + ((1 - alpha) * avg_content_score)
            hybrid_scores.append((movie_id, hybrid_score))
        except (IndexError, ValueError): continue
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    recs = []
    for movie_id, score in hybrid_scores[:n]:
        recs.append({
            "movie_id": int(movie_id), "title": movie_id_to_title.get(movie_id, ""),
            "hybrid_score": round(score, 4),
            "explanation": get_content_explanation(user_id, movie_id)
        })
    return add_metadata_to_recs(recs)

def get_sequential_recommendations(user_id, n=10):
    if user_id not in sequences: return []
    user_sequence_original_ids = sequences[user_id]
    user_sequence_mapped = [movie_id_map_seq[mid] for mid in user_sequence_original_ids if mid in movie_id_map_seq]
    if not user_sequence_mapped: return []
    input_tensor = torch.tensor([user_sequence_mapped], dtype=torch.long)
    with torch.no_grad(): output, _ = sequential_model(input_tensor)
    top_scores, top_indices = torch.topk(output, k=n + len(user_sequence_mapped))
    recs = []
    for idx in top_indices.squeeze().tolist():
        movie_id = idx_to_movie_id_seq.get(idx)
        if movie_id and movie_id not in user_sequence_original_ids:
            recs.append({"movie_id": int(movie_id), "title": movie_id_to_title.get(movie_id, "")})
        if len(recs) == n: break
    return recs

def get_autoencoder_recommendations(user_id, n=10):
    if user_id not in user_item_matrix.index: return []
    user_vector = torch.FloatTensor(user_item_matrix.loc[user_id].values).unsqueeze(0)
    with torch.no_grad(): reconstructed_vector = autoencoder_model(user_vector)
    predicted_ratings = reconstructed_vector.squeeze().numpy() * 4 + 1
    unrated_mask = user_vector.squeeze().numpy() == 0
    scores_for_unrated = predicted_ratings[unrated_mask]
    original_indices_of_unrated = np.where(unrated_mask)[0]
    top_n_indices_in_scores = scores_for_unrated.argsort()[-n:][::-1]
    recs = []
    for i in top_n_indices_in_scores:
        original_matrix_idx = original_indices_of_unrated[i]
        predicted_score = scores_for_unrated[i]
        movie_id = idx_to_movie_id_ae.get(original_matrix_idx)
        if movie_id:
            recs.append({
                "movie_id": int(movie_id), "title": movie_id_to_title.get(movie_id, ""),
                "predicted_rating": round(float(predicted_score), 4)
            })
    return recs


# --- 6. API Endpoints ---
@app.get("/")
def read_root(): return {"message": "Welcome to the MoviePulse API"}

@app.get("/homepage/{user_id}")
def get_homepage(user_id: int):
    print(f"Generating homepage for user {user_id}")
    svd_recs_pool = get_top_n_recommendations(user_id, n=100)
    
    hero_rec = svd_recs_pool[0] if svd_recs_pool else {}
    if hero_rec: add_metadata_to_recs([hero_rec])

    shelves = []
    for_you_recs = add_metadata_to_recs(svd_recs_pool[1:11])
    shelves.append({"title": "Personalized For You", "movies": for_you_recs})

    sequential_recs = add_metadata_to_recs(get_sequential_recommendations(user_id, n=10))
    if sequential_recs: shelves.append({"title": "What to Watch Next", "movies": sequential_recs})

    action_recs = add_metadata_to_recs([r for r in svd_recs_pool if "Action" in movie_id_to_genres.get(r['movie_id'], "")][:10])
    if action_recs: shelves.append({"title": "Action & Adventure", "movies": action_recs})
        
    comedy_recs = add_metadata_to_recs([r for r in svd_recs_pool if "Comedy" in movie_id_to_genres.get(r['movie_id'], "")][:10])
    if comedy_recs: shelves.append({"title": "Top Comedies", "movies": comedy_recs})

    return {"hero": hero_rec, "shelves": shelves}

@app.get("/recommend/user/{user_id}")
def get_user_recommendations(user_id: int, count: int = 10):
    return {"user_id": user_id, "recommendations": add_metadata_to_recs(get_top_n_recommendations(user_id, n=count))}

@app.get("/recommend/hybrid/{user_id}")
def get_hybrid_user_recommendations(user_id: int, count: int = 10):
    return {"user_id": user_id, "recommendations": get_hybrid_recommendations(user_id, n=count)}

@app.get("/recommend/sequential/{user_id}")
def get_next_item_recommendations(user_id: int, count: int = 10):
    return {"user_id": user_id, "recommendations": add_metadata_to_recs(get_sequential_recommendations(user_id, n=count))}

@app.get("/recommend/autoencoder/{user_id}")
def get_ae_recommendations(user_id: int, count: int = 10):
    return {"user_id": user_id, "recommendations": add_metadata_to_recs(get_autoencoder_recommendations(user_id, n=count))}

@app.post("/feedback")
def process_feedback(feedback: Feedback):
    print(f"Feedback Received: User {feedback.user_id} '{feedback.feedback_type}d' Movie {feedback.movie_id}")
    return {"status": "success", "message": "Feedback received."}


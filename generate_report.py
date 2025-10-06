# generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

print("--- Generating MoviePulse Analytics Report ---")

# --- 1. Load and Prepare Data ---
# Load ratings data
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Load movie data
i_cols = [
    'movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 
    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# Merge for richer analysis
data = pd.merge(ratings, movies, on='movie_id')


# --- 2. Perform Analysis & Calculate KPIs ---
# Overall Stats
total_users = data['user_id'].nunique()
total_movies = data['movie_id'].nunique()
total_ratings = len(data)

# Top 10 Most Rated Movies
top_10_movies = data['title'].value_counts().head(10).reset_index()
top_10_movies.columns = ['Movie Title', 'Number of Ratings']

# Ratings Over Time
ratings_per_month = ratings.set_index('date').resample('M').size().reset_index(name='count')
ratings_per_month['month'] = ratings_per_month['date'].dt.strftime('%Y-%m')

# Genre Distribution
genre_cols = i_cols[6:]
genre_counts = movies[genre_cols].sum().sort_values(ascending=False)


# --- 3. Generate Plots and Encode for HTML ---
def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Plot 1: Ratings per Month
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(ratings_per_month['month'], ratings_per_month['count'], color='#007acc')
ax1.set_title('User Engagement: Ratings per Month')
ax1.set_ylabel('Number of Ratings')
ax1.tick_params(axis='x', rotation=45)
plot1_base64 = fig_to_base64(fig1)
plt.close(fig1)

# Plot 2: Genre Distribution
fig2, ax2 = plt.subplots(figsize=(10, 5))
genre_counts.plot(kind='bar', ax=ax2, color='#34a853')
ax2.set_title('Content: Movie Distribution by Genre')
ax2.set_ylabel('Number of Movies')
ax2.tick_params(axis='x', rotation=90)
plot2_base64 = fig_to_base64(fig2)
plt.close(fig2)


# --- 4. Generate the HTML Report ---
html_template = f"""
<html>
<head>
    <title>MoviePulse Analytics Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        h1, h2 {{ color: #333; }}
        .kpi-container {{ display: flex; justify-content: space-around; background-color: #f2f2f2; padding: 1em; border-radius: 8px; }}
        .kpi {{ text-align: center; }}
        .kpi .value {{ font-size: 2em; font-weight: bold; }}
        .kpi .label {{ color: #666; }}
        .content-container {{ display: flex; justify-content: space-around; margin-top: 2em; flex-wrap: wrap; }}
        .table-container, .plot-container {{ width: 48%; margin-bottom: 2em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <h1>🎬 MoviePulse Analytics Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Overall System Health</h2>
    <div class="kpi-container">
        <div class="kpi"><div class="value">{total_users}</div><div class="label">Total Users</div></div>
        <div class="kpi"><div class="value">{total_movies}</div><div class="label">Total Movies</div></div>
        <div class="kpi"><div class="value">{total_ratings}</div><div class="label">Total Ratings</div></div>
    </div>

    <div class="content-container">
        <div class="table-container">
            <h2>🏆 Top 10 Most Rated Movies</h2>
            {top_10_movies.to_html(index=False)}
        </div>
        <div class="plot-container">
            <h2>📈 User Engagement Over Time</h2>
            <img src="data:image/png;base64,{plot1_base64}">
        </div>
    </div>

    <div class="content-container">
        <div class="plot-container" style="width:100%">
            <h2>🎭 Content Catalog: Genre Distribution</h2>
            <img src="data:image/png;base64,{plot2_base64}">
        </div>
    </div>
</body>
</html>
"""

# Save the HTML to a file
with open('report.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("\nSuccess! Report saved to 'report.html'. Open this file in your browser.")
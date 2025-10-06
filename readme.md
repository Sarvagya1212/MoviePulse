🎬 MoviePulse: A Netflix-Style Recommendation System
MoviePulse is a full-stack, AI-powered movie recommendation application designed to replicate the sophisticated, user-centric experience of modern streaming platforms. It leverages a powerful multi-model Python backend to generate deeply personalized recommendations and serves them through a dynamic, fully responsive React frontend.

This project is more than just an algorithm; it's a complete, end-to-end prototype that demonstrates the entire lifecycle of building a modern recommendation system, from data analysis and model training to deployment architecture and user interface design.

<!-- It's best to replace this with a real screenshot of your app -->

✨ Features
MoviePulse is packed with features that showcase a production-ready approach to recommendation systems.

Dual-View Interface: The application provides two distinct user experiences, switchable from the header:

Homepage View: A polished, user-facing layout designed for an immersive discovery experience. It features a large "hero" banner for the top recommendation and horizontally scrolling "shelves" for different categories, mimicking the familiar interface of services like Netflix.

Model Explorer View: A technical dashboard designed for demonstration and analysis. It allows you to directly select and compare the raw outputs of the different recommendation algorithms, providing insight into how each model behaves for a given user.

Multi-Model Recommendation Engine: At its core, MoviePulse is not a single algorithm but an orchestrated system of multiple specialized models, each solving a different part of the recommendation problem:

SVD (Collaborative Filtering): The primary engine for the "For You" shelf. After rigorous offline evaluation, a tuned SVD model proved to be the most accurate at finding users with similar tastes and predicting ratings.

Sequential (GRU): A deep learning model that powers the "What to Watch Next" suggestions. Unlike other models, this Gated Recurrent Unit analyzes the order of a user's viewing history, making its recommendations feel more timely and context-aware.

Autoencoder: A neural network that learns a compressed "taste profile" for each user, allowing it to find users with similar latent preferences and recommend surprising, novel content.

Hybrid Model: The "Smart Picks" algorithm combines the strengths of collaborative filtering (what similar users like) and content-based filtering (what makes movies similar) to provide recommendations that are both accurate and explainable.

Rich Movie Data & Caching: The system integrates with the TMDb (The Movie Database) API in real-time to enrich recommendations with high-quality movie posters, backdrops, plot summaries, and genres. To ensure performance and avoid rate limiting, this metadata is cached in the backend after the first lookup.

Interactive Feedback Loop: The UI is not a one-way street. Users can provide explicit "like" or "pass" feedback on any recommendation. This interaction is sent back to the API, creating a valuable data stream that, in a production environment, would be logged and used to continuously retrain and improve the models over time.

Fully Responsive: The user interface is built with a mobile-first approach using Tailwind CSS, ensuring that the application is beautiful, functional, and easy to use on all devices, from a small mobile phone to a large desktop monitor.

🛠️ Technology Stack
The technologies were chosen to create a modern, high-performance, and scalable application.

Backend:

Python: The language of choice for data science and machine learning.

FastAPI: A high-performance web framework for building the API, chosen for its speed, asynchronous capabilities, and automatic documentation.

Pandas & NumPy: For efficient data manipulation and numerical computation.

Surprise & Scikit-learn: For implementing classical machine learning models like SVD and content-based filtering.

PyTorch: For building and training the deep learning models (GRU and Autoencoder).

Frontend:

React: A powerful JavaScript library for building component-based, interactive user interfaces.

Vite: A next-generation frontend tooling that provides an extremely fast development server and optimized production builds.

Tailwind CSS: A utility-first CSS framework for rapidly building modern and responsive designs without writing custom CSS.

Lucide React: For clean, lightweight, and consistent icons throughout the application.

External APIs: TMDb (The Movie Database) for enriching movie metadata.

🚀 Getting Started
Follow these instructions to get the MoviePulse application running on your local machine.

Prerequisites
Make sure you have the following software installed on your system:

Python (3.8 or higher)

Node.js and npm (v16 or higher)

Git for cloning the repository.

1. Clone the Repository
First, clone the project repository to your local machine using your preferred method:

git clone <your-repository-url>
cd moviepulse

2. Set Up the Python Backend
The backend server is the application's brain, powering all the machine learning models and data fetching.

# 1. Create and activate a Python virtual environment. This isolates the project's dependencies.
python -m venv .venv
source .venv/bin/activate  # On Windows, use the command: .venv\Scripts\activate

# 2. Install all the required Python packages from the requirements file.
pip install -r requirements.txt

# 3. (Optional but Highly Recommended) Add your TMDb API Key
#    - Sign up for a free API key at themoviedb.org.
#    - Open the `main.py` file and find the line `TMDB_API_KEY = "YOUR_TMDB_API_KEY"`.
#    - Replace "YOUR_TMDB_API_KEY" with your actual key. This is required for movie posters and details to appear.

3. Set Up the React Frontend
The frontend is the user interface you will interact with in the browser.

# 1. Navigate into the frontend directory from the project root.
cd frontend

# 2. Install all the required npm packages defined in package.json.
npm install

4. Run the Application
To run MoviePulse, you need to have two separate terminals open simultaneously, one for the backend and one for the frontend.

Terminal 1: Start the Backend API

Make sure you are in the main moviepulse directory (the project root).

# Activate the virtual environment if it's not already active in this terminal.
source .venv/bin/activate # or .venv\Scripts\activate on Windows

# Start the FastAPI server with hot-reloading enabled.
uvicorn main:app --reload

Your API will now be running and accessible at http://127.0.0.1:8000.

Terminal 2: Start the Frontend App

Make sure you are in the frontend directory.

# Start the Vite React development server.
npm run dev

Your React application will now be running, typically at http://localhost:5173.

You can now open http://localhost:5173 in your web browser to use the MoviePulse application!

📂 Project Structure
The project is organized into a clean, standard structure for a full-stack application.

moviepulse/
│
├── .venv/                  # Contains the Python virtual environment interpreters and packages.
├── frontend/               # The complete React frontend application.
│   ├── src/
│   │   ├── App.jsx         # The main, unified React component that controls the entire UI.
│   │   └── index.css     # The entry point for Tailwind CSS styles.
│   ├── index.html          # The HTML entry point for the Vite development server.
│   └── package.json        # Defines frontend dependencies and scripts (like `npm run dev`).
│
├── ml-100k/                # The MovieLens 100k dataset (should be downloaded and placed here).
├── .gitignore              # Specifies files and folders for Git to ignore (e.g., .venv, *.pkl).
├── main.py                 # The main Python FastAPI backend server script.
├── README.md               # This documentation file.
├── requirements.txt        # Lists all Python dependencies for the backend.
└── *.pkl                   # The saved, pre-trained machine learning models.
# MoviePulse

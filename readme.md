# 🎬 MoviePulse

A Netflix-style AI-powered movie recommendation system built with FastAPI, React, and multiple machine learning models.

![MoviePulse Banner](https://via.placeholder.com/1200x400/1a1a2e/00d9ff?text=MoviePulse)

## 📖 Overview

MoviePulse is a full-stack movie recommendation application that replicates the sophisticated, user-centric experience of modern streaming platforms. It leverages a powerful multi-model Python backend to generate deeply personalized recommendations and serves them through a dynamic, fully responsive React frontend.

This project demonstrates the complete lifecycle of building a production-ready recommendation system, from data analysis and model training to deployment architecture and user interface design.

## ✨ Features

### 🎯 Dual-View Interface
- **Homepage View**: Polished, user-facing layout with hero banners and horizontally scrolling shelves mimicking Netflix
- **Model Explorer View**: Technical dashboard for comparing raw outputs of different recommendation algorithms

### 🤖 Multi-Model Recommendation Engine

| Model | Purpose | Technology |
|-------|---------|------------|
| **SVD** | Collaborative filtering for "For You" recommendations | Surprise library |
| **Sequential GRU** | Context-aware "What to Watch Next" based on viewing order | PyTorch |
| **Autoencoder** | Novel content discovery through latent taste profiles | PyTorch |
| **Hybrid Model** | "Smart Picks" combining collaborative + content-based filtering | Scikit-learn |

### 🎨 Additional Features
- **Real-time TMDb Integration**: High-quality posters, backdrops, and metadata
- **Smart Caching**: Backend caching to prevent rate limiting and improve performance
- **Interactive Feedback**: Like/pass system for continuous model improvement
- **Fully Responsive**: Mobile-first design with Tailwind CSS

## 🛠️ Technology Stack

### Backend
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

- **FastAPI** - High-performance async web framework
- **Pandas & NumPy** - Data manipulation and computation
- **Surprise** - Collaborative filtering algorithms
- **Scikit-learn** - Classical ML models
- **PyTorch** - Deep learning (GRU, Autoencoder)

### Frontend
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-38B2AC?logo=tailwind-css&logoColor=white)

- **React** - Component-based UI library
- **Vite** - Next-generation frontend tooling
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Lightweight icon library

### External APIs
- **TMDb API** - Movie metadata and imagery

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Node.js and npm (v16+)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd moviepulse
   ```

2. **Set up the Python backend**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure TMDb API (Optional but Recommended)**
   - Sign up for a free API key at [themoviedb.org](https://www.themoviedb.org/)
   - Open `main.py` and replace:
     ```python
     TMDB_API_KEY = "YOUR_TMDB_API_KEY"
     ```

4. **Set up the React frontend**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

You'll need two terminal windows:

**Terminal 1 - Backend:**
```bash
# From project root
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uvicorn main:app --reload
```
Backend will run at `http://127.0.0.1:8000`

**Terminal 2 - Frontend:**
```bash
# From frontend directory
cd frontend
npm run dev
```
Frontend will run at `http://localhost:5173`

Open `http://localhost:5173` in your browser to start using MoviePulse! 🎉

## 📂 Project Structure

```
moviepulse/
├── .venv/                  # Python virtual environment
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   └── index.css      # Tailwind CSS entry point
│   ├── index.html         # HTML entry point
│   └── package.json       # Frontend dependencies
├── ml-100k/               # MovieLens 100k dataset
├── main.py                # FastAPI backend server
├── requirements.txt       # Python dependencies
├── *.pkl                  # Pre-trained ML models
├── .gitignore
└── README.md
```

## 🎯 Usage

### Homepage View
- Browse personalized recommendations in different categories
- Click on movie cards to view details
- Use like/pass buttons to provide feedback

### Model Explorer View
- Switch views using the header navigation
- Select different recommendation models from the dropdown
- Compare outputs to understand model behavior
- Analyze how each algorithm generates recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [TMDb](https://www.themoviedb.org/) for movie metadata and imagery
- The open-source community for the amazing libraries and tools

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/moviepulse](https://github.com/yourusername/moviepulse)

---

<p align="center">Made with ❤️ and lots of ☕</p>

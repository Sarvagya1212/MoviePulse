import React, { useState, useEffect } from 'react';
import { Play, Info, ThumbsUp, ThumbsDown } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

// --- Components ---

function Hero({ movie }) {
    if (!movie) return <div className="h-[56.25vw] max-h-[700px] bg-gray-800 animate-pulse"></div>;
    
    const backdropUrl = movie.backdrop_url || movie.poster_url;

    return (
        <div className="relative h-[56.25vw] max-h-[700px] w-full text-white">
            {backdropUrl && <img src={backdropUrl} alt={movie.title} className="absolute inset-0 w-full h-full object-cover" />}
            <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/60 to-transparent"></div>
            <div className="absolute bottom-0 left-0 p-4 md:p-12 max-w-2xl">
                <h1 className="text-3xl md:text-6xl font-bold drop-shadow-lg">{movie.title}</h1>
                <p className="mt-4 text-sm md:text-base line-clamp-3 drop-shadow-md">{movie.overview}</p>
                <div className="mt-6 flex space-x-4">
                    <button className="flex items-center space-x-2 bg-white text-black px-6 py-2 rounded font-semibold hover:bg-gray-200 transition">
                        <Play className="w-6 h-6 fill-current" />
                        <span>Play</span>
                    </button>
                    <button className="flex items-center space-x-2 bg-gray-500/50 text-white px-6 py-2 rounded font-semibold hover:bg-gray-500/70 transition backdrop-blur-sm">
                        <Info className="w-6 h-6" />
                        <span>More Info</span>
                    </button>
                </div>
            </div>
        </div>
    );
}

function MovieCard({ movie }) {
    return (
        <div className="group relative w-40 md:w-60 flex-shrink-0 rounded-md overflow-hidden transform transition-transform duration-300 hover:scale-110 hover:z-10">
            {movie.poster_url ? (
                <img src={movie.poster_url} alt={movie.title} className="w-full h-full object-cover" />
            ) : (
                <div className="w-full h-full bg-gray-800 flex items-center justify-center text-gray-500">{movie.title}</div>
            )}
            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity p-4 flex flex-col justify-end">
                <h3 className="text-white font-bold">{movie.title}</h3>
                <p className="text-xs text-gray-300 mt-1">{movie.genres}</p>
                 <div className="flex space-x-2 mt-4">
                    <button className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center hover:bg-white/40"><ThumbsUp className="w-4 h-4" /></button>
                    <button className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center hover:bg-white/40"><ThumbsDown className="w-4 h-4" /></button>
                </div>
            </div>
        </div>
    );
}

function MovieRow({ title, movies }) {
    return (
        <div className="my-8">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-4 px-4 md:px-12">{title}</h2>
            <div className="flex space-x-4 overflow-x-auto pb-4 px-4 md:px-12 scrollbar-hide">
                {movies.map(movie => <MovieCard key={movie.movie_id} movie={movie} />)}
            </div>
        </div>
    );
}

// --- Main App Component ---

export default function NetflixUI() {
    const [userId, setUserId] = useState('196'); // Default user
    const [homepageData, setHomepageData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchHomepage = async () => {
            setLoading(true);
            setError('');
            try {
                const response = await fetch(`${API_BASE_URL}/homepage/${userId}`);
                if (!response.ok) throw new Error('Failed to fetch homepage data');
                const data = await response.json();
                setHomepageData(data);
            } catch (err) {
                setError('Could not connect to the API. Is the server running?');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchHomepage();
    }, [userId]);

    return (
        <div className="bg-slate-900 min-h-screen text-white">
            <style>{`.scrollbar-hide::-webkit-scrollbar { display: none; } .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }`}</style>
            
            {loading && <div className="flex items-center justify-center h-screen"><div className="w-16 h-16 border-4 border-t-red-600 border-gray-700 rounded-full animate-spin"></div></div>}
            {error && <div className="flex items-center justify-center h-screen text-red-400">{error}</div>}
            
            {!loading && !error && homepageData && (
                <div>
                    <Hero movie={homepageData.hero} />
                    <div className="relative -top-16">
                        {homepageData.shelves.map(shelf => (
                            <MovieRow key={shelf.title} title={shelf.title} movies={shelf.movies} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}


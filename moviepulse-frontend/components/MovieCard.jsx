import React, { useState } from 'react';
import { Film, ThumbsUp, ThumbsDown, Star, Play } from 'lucide-react';

export function MovieCard({ movie, onSelectMovie, onFeedback }) {
    const [feedbackStatus, setFeedbackStatus] = useState(null);

    const handleFeedback = (e, feedbackType) => {
        e.stopPropagation(); // Prevent the modal from opening
        setFeedbackStatus(feedbackType);
        onFeedback(movie.movie_id, feedbackType);
        setTimeout(() => setFeedbackStatus(null), 2000);
    };

    const score = movie.estimated_rating || movie.hybrid_score || movie.predicted_rating;

    return (
        <div
            className="group bg-black/40 backdrop-blur-sm border border-purple-500/20 rounded-xl overflow-hidden hover:border-purple-500/60 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/30 hover:-translate-y-2 cursor-pointer"
            onClick={() => onSelectMovie(movie)}
        >
            <div className="relative aspect-[2/3] overflow-hidden bg-slate-800">
                {movie.poster_url ? (
                    <img src={movie.poster_url} alt={movie.title} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" loading="lazy" />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                        <Film className="w-20 h-20 text-slate-600" />
                    </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                {score && (
                    <div className="absolute top-3 right-3 bg-gradient-to-br from-yellow-500 to-orange-500 text-white px-3 py-1.5 rounded-full text-sm font-bold shadow-lg flex items-center space-x-1 backdrop-blur-sm">
                        <Star className="w-3.5 h-3.5 fill-current" />
                        <span>{score.toFixed(1)}</span>
                    </div>
                )}
                 <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-white/90 hover:bg-white text-purple-900 px-4 py-2 rounded-lg font-semibold opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center space-x-2 shadow-lg transform translate-y-2 group-hover:translate-y-0">
                    <Play className="w-4 h-4" />
                    <span>View Details</span>
                </div>
            </div>
            <div className="p-4">
                <h3 className="text-white font-semibold mb-2 line-clamp-2 group-hover:text-purple-300 transition-colors leading-snug">
                    {movie.title}
                </h3>
                <div className="flex items-center space-x-2">
                    <button
                        onClick={(e) => handleFeedback(e, 'like')}
                        className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-2.5 rounded-lg font-medium transition-all duration-200 text-sm ${
                            feedbackStatus === 'like'
                                ? 'bg-green-500 text-white shadow-lg shadow-green-500/30 scale-105'
                                : 'bg-white/5 text-slate-400 hover:bg-green-500/20 hover:text-green-400 border border-slate-700/50 hover:border-green-500/50'
                        }`}
                    >
                        <ThumbsUp className="w-4 h-4" />
                        <span>Like</span>
                    </button>
                    <button
                        onClick={(e) => handleFeedback(e, 'dislike')}
                         className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-2.5 rounded-lg font-medium transition-all duration-200 text-sm ${
                            feedbackStatus === 'dislike'
                                ? 'bg-red-500 text-white shadow-lg shadow-red-500/30 scale-105'
                                : 'bg-white/5 text-slate-400 hover:bg-red-500/20 hover:text-red-400 border border-slate-700/50 hover:border-red-500/50'
                        }`}
                    >
                        <ThumbsDown className="w-4 h-4" />
                        <span>Pass</span>
                    </button>
                </div>
            </div>
        </div>
    );
}


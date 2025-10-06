import React, { useState, useEffect } from 'react';
import { Film, Sparkles, User, TrendingUp, Zap, Search, Info, X, RefreshCw, Star, Play, ThumbsUp, ThumbsDown, LayoutGrid, Home } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

// --- Reusable API Functions ---
const api = {
    fetchHomepage: async (userId) => {
        const response = await fetch(`${API_BASE_URL}/homepage/${userId}`);
        if (!response.ok) throw new Error('Failed to fetch homepage data');
        const data = await response.json();
        return data;
    },
    fetchRecommendations: async (userId, modelEndpoint) => {
        const response = await fetch(`${API_BASE_URL}/recommend/${modelEndpoint}/${userId}?count=8`);
        if (!response.ok) throw new Error('Failed to fetch model recommendations');
        const data = await response.json();
        return data.recommendations || [];
    },
    sendFeedback: async (userId, movieId, feedbackType) => {
        await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: parseInt(userId), movie_id: movieId, feedback_type: feedbackType })
        });
    }
};

// --- Child Components ---

function MovieModal({ movie, onClose }) {
    if (!movie) return null;
    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in" onClick={onClose}>
            <div className="bg-slate-900 border border-purple-500/30 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl shadow-purple-500/20" onClick={(e) => e.stopPropagation()}>
                <div className="relative">
                    {movie.poster_url && (
                        <div className="relative h-60 sm:h-80 overflow-hidden rounded-t-2xl">
                            <img src={movie.poster_url} alt={movie.title} className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/50 to-transparent"></div>
                        </div>
                    )}
                    <button onClick={onClose} className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-all backdrop-blur-sm"><X className="w-5 h-5" /></button>
                </div>
                <div className="p-4 sm:p-6">
                    <h2 className="text-xl sm:text-2xl font-bold text-white mb-3">{movie.title}</h2>
                    {movie.genres && (
                        <div className="flex flex-wrap gap-2 mb-4">
                            {movie.genres.split(' | ').map((genre, i) => (<span key={i} className="bg-purple-600/30 text-purple-200 text-xs sm:text-sm px-3 py-1 rounded-full border border-purple-500/40">{genre}</span>))}
                        </div>
                    )}
                     <p className="text-slate-300 leading-relaxed text-sm sm:text-base">{movie.overview || "No overview available."}</p>
                    {movie.explanation && (
                         <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 my-4"><p className="text-purple-300 text-sm italic">{movie.explanation}</p></div>
                    )}
                </div>
            </div>
        </div>
    );
}

function Hero({ movie }) {
    if (!movie || !movie.title) return <div className="h-[70vh] sm:h-[56.25vw] max-h-[700px] bg-slate-800 animate-pulse"></div>;
    const backdropUrl = movie.backdrop_url || movie.poster_url;
    return (
        <div className="relative h-[70vh] sm:h-[56.25vw] max-h-[700px] w-full text-white">
            {backdropUrl && <img src={backdropUrl} alt={movie.title} className="absolute inset-0 w-full h-full object-cover" />}
            <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/60 to-transparent"></div>
            <div className="absolute bottom-0 left-0 p-4 md:p-12 max-w-2xl">
                <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold drop-shadow-lg">{movie.title}</h1>
                <p className="mt-4 text-sm md:text-base line-clamp-3 drop-shadow-md">{movie.overview}</p>
                <div className="mt-6 flex space-x-4">
                    <button className="flex items-center space-x-2 bg-white text-black px-4 sm:px-6 py-2 rounded font-semibold hover:bg-gray-200 transition text-sm sm:text-base"><Play className="w-5 h-5 sm:w-6 sm:h-6 fill-current" /><span>Play</span></button>
                    <button className="flex items-center space-x-2 bg-gray-500/50 text-white px-4 sm:px-6 py-2 rounded font-semibold hover:bg-gray-500/70 transition backdrop-blur-sm text-sm sm:text-base"><Info className="w-5 h-5 sm:w-6 sm:h-6" /><span>More Info</span></button>
                </div>
            </div>
        </div>
    );
}

function HomepageMovieCard({ movie, onSelectMovie }) {
    return (
        <div onClick={() => onSelectMovie(movie)} className="group relative w-40 md:w-60 flex-shrink-0 rounded-md overflow-hidden transform transition-transform duration-300 hover:scale-110 hover:z-10 cursor-pointer">
            {movie.poster_url ? <img src={movie.poster_url} alt={movie.title} className="w-full h-full object-cover" /> : <div className="w-full h-full bg-gray-800 flex items-center justify-center text-gray-500 text-center p-2">{movie.title}</div>}
            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity p-4 flex flex-col justify-end">
                <h3 className="text-white font-bold">{movie.title}</h3>
                <p className="text-xs text-gray-300 mt-1 line-clamp-2">{movie.genres}</p>
            </div>
        </div>
    );
}

function MovieRow({ title, movies, onSelectMovie }) {
    return (
        <div className="my-8">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-4 px-4 md:px-12">{title}</h2>
            <div className="flex space-x-4 overflow-x-auto pb-4 px-4 md:px-12 scrollbar-hide">
                {movies.map(movie => <HomepageMovieCard key={movie.movie_id} movie={movie} onSelectMovie={onSelectMovie} />)}
            </div>
        </div>
    );
}

function HomepageView({ loading, error, data, onSelectMovie }) {
    if (loading) return <div className="flex items-center justify-center h-screen"><div className="w-16 h-16 border-4 border-t-red-600 border-gray-700 rounded-full animate-spin"></div></div>;
    if (error) return <div className="flex items-center justify-center h-screen text-red-400 p-4 text-center">{error}</div>;
    if (!data) return null;

    return (
        <div>
            <Hero movie={data.hero} />
            <div className="relative md:-top-16">
                {data.shelves.map(shelf => <MovieRow key={shelf.title} title={shelf.title} movies={shelf.movies} onSelectMovie={onSelectMovie} />)}
            </div>
        </div>
    );
}

function ExplorerMovieCard({ movie, onFeedback, feedbackStatus, onSelectMovie }) {
     const score = movie.estimated_rating || movie.hybrid_score || movie.predicted_rating;
    return (
        <div className="group bg-black/40 backdrop-blur-sm border border-purple-500/20 rounded-xl overflow-hidden hover:border-purple-500/60 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/30 hover:-translate-y-2">
            <div className="relative aspect-[2/3] overflow-hidden bg-slate-800">
                {movie.poster_url ? <img src={movie.poster_url} alt={movie.title} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" loading="lazy" /> : <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900"><Film className="w-20 h-20 text-slate-600" /></div>}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                {score && <div className="absolute top-3 right-3 bg-gradient-to-br from-yellow-500 to-orange-500 text-white px-3 py-1.5 rounded-full text-sm font-bold shadow-lg flex items-center space-x-1 backdrop-blur-sm"><Star className="w-3.5 h-3.5 fill-current" /><span>{score.toFixed(1)}</span></div>}
                <button onClick={() => onSelectMovie(movie)} className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-white/90 hover:bg-white text-purple-900 px-4 py-2 rounded-lg font-semibold opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center space-x-2 shadow-lg transform translate-y-2 group-hover:translate-y-0"><Play className="w-4 h-4" /><span>View Details</span></button>
            </div>
            <div className="p-4">
                <h3 className="text-white font-semibold mb-2 line-clamp-2 group-hover:text-purple-300 transition-colors leading-snug">{movie.title}</h3>
                {movie.explanation && <p className="text-purple-300/80 text-xs mb-3 italic line-clamp-2 bg-purple-500/10 p-2 rounded-lg border border-purple-500/20">{movie.explanation}</p>}
                <div className="flex items-center space-x-2">
                    <button onClick={() => onFeedback(movie.movie_id, 'like')} className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-2.5 rounded-lg font-medium transition-all duration-200 text-sm ${feedbackStatus[movie.movie_id] === 'like' ? 'bg-green-500 text-white shadow-lg shadow-green-500/30 scale-105' : 'bg-white/5 text-slate-400 hover:bg-green-500/20 hover:text-green-400 border border-slate-700/50 hover:border-green-500/50'}`}><ThumbsUp className="w-4 h-4" /><span>Like</span></button>
                    <button onClick={() => onFeedback(movie.movie_id, 'dislike')} className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-2.5 rounded-lg font-medium transition-all duration-200 text-sm ${feedbackStatus[movie.movie_id] === 'dislike' ? 'bg-red-500 text-white shadow-lg shadow-red-500/30 scale-105' : 'bg-white/5 text-slate-400 hover:bg-red-500/20 hover:text-red-400 border border-slate-700/50 hover:border-red-500/50'}`}><ThumbsDown className="w-4 h-4" /><span>Pass</span></button>
                </div>
            </div>
        </div>
    );
}

function ModelExplorerView({ userId, loading, error, recommendations, onFeedback, onSelectMovie, activeTab, onTabChange, onRefresh, feedbackStatus }) {
    const tabs = [
        { id: 'hybrid', name: 'Smart Picks', icon: Sparkles, desc: 'Combines your preferences with content similarity' },
        { id: 'user', name: 'For You', icon: User, desc: 'Personalized based on your rating history' },
        { id: 'sequential', name: 'Next Watch', icon: TrendingUp, desc: 'Based on your recent viewing patterns' },
        { id: 'autoencoder', name: 'AI Picks', icon: Zap, desc: 'Deep learning predictions for your taste' }
    ];

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="mb-8">
                <div className="flex items-center justify-between mb-4"><h2 className="text-xl font-semibold text-white">Choose Recommendation Type</h2><button onClick={onRefresh} disabled={loading} className="flex items-center space-x-2 bg-white/10 hover:bg-white/20 text-purple-300 px-4 py-2 rounded-lg transition-all border border-purple-500/30 disabled:opacity-50"><RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /><span>Refresh</span></button></div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                    {tabs.map((tab) => <button key={tab.id} onClick={() => onTabChange(tab.id)} className={`group relative p-4 rounded-xl font-medium transition-all duration-300 text-left overflow-hidden ${activeTab === tab.id ? 'bg-gradient-to-br from-purple-600 to-pink-600 text-white shadow-xl scale-105' : 'bg-black/30 text-purple-300 hover:bg-black/40 border border-purple-500/20'}`}><div className="relative z-10"><div className="flex items-center space-x-3 mb-2"><tab.icon className="w-6 h-6" /> <span className="font-semibold">{tab.name}</span></div><p className={`text-xs leading-relaxed ${activeTab === tab.id ? 'text-purple-100' : 'text-purple-400'}`}>{tab.desc}</p></div></button>)}
                </div>
            </div>

            {loading && <div className="flex flex-col items-center justify-center py-24"><div className="w-20 h-20 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div><p className="text-purple-300 text-lg mt-6 font-medium">Curating recommendations...</p></div>}
            {error && <div className="mb-8 bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-300 flex items-start space-x-3"><X className="w-5 h-5 mt-0.5" /><div><p className="font-semibold">Error</p><p className="text-sm">{error}</p></div></div>}
            
            {!loading && recommendations.length > 0 && (
                 <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                    {recommendations.map(movie => <ExplorerMovieCard key={movie.movie_id} movie={movie} onFeedback={onFeedback} onSelectMovie={onSelectMovie} feedbackStatus={feedbackStatus} />)}
                </div>
            )}
             {!loading && !userId && !error && (
                <div className="text-center py-24"><div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 p-12 rounded-3xl border border-purple-500/30 inline-block"><Film className="w-24 h-24 text-purple-400" /></div><h2 className="mt-8 text-4xl font-bold text-white">Welcome to MoviePulse</h2><p className="mt-4 text-purple-300 text-lg">Enter a User ID above to get started.</p></div>
            )}
        </div>
    );
}

// --- The MAIN Application Shell ---

export default function App() {
    const [view, setView] = useState('homepage');
    const [userId, setUserId] = useState('196');
    const [searchUserId, setSearchUserId] = useState('196');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [selectedMovie, setSelectedMovie] = useState(null);
    const [feedbackStatus, setFeedbackStatus] = useState({});
    const [homepageData, setHomepageData] = useState(null);
    const [activeTab, setActiveTab] = useState('hybrid');
    const [explorerRecs, setExplorerRecs] = useState([]);
    
    const tabs = [
        { id: 'hybrid', name: 'Smart Picks', icon: Sparkles, endpoint: 'hybrid' },
        { id: 'user', name: 'For You', icon: User, endpoint: 'user' },
        { id: 'sequential', name: 'Next Watch', icon: TrendingUp, endpoint: 'sequential' },
        { id: 'autoencoder', name: 'AI Picks', icon: Zap, endpoint: 'autoencoder' }
    ];
    
    const handleFetch = async (uid, tab = activeTab) => {
        if (!uid || uid < 1 || uid > 943) { setError('Please enter a valid User ID (1-943)'); setLoading(false); return; }
        setLoading(true); setError(''); 
        
        try {
            if (view === 'homepage') {
                setHomepageData(null);
                const data = await api.fetchHomepage(uid);
                setHomepageData(data);
            } else {
                setExplorerRecs([]);
                const endpoint = tabs.find(t=>t.id === tab)?.endpoint || 'hybrid';
                const data = await api.fetchRecommendations(uid, endpoint);
                setExplorerRecs(data);
            }
            setUserId(uid);
        } catch(err) {
            setError('Unable to connect to the API. Make sure the server is running on port 8000.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };
    
    useEffect(() => {
        handleFetch(userId, activeTab);
    }, [view, userId, activeTab]);

    const handleSearch = () => {
        setUserId(searchUserId);
    };
    
    const handleTabChange = (tabId) => {
        setActiveTab(tabId);
    };

    const handleFeedback = (movieId, feedbackType) => {
        if (!userId) return;
        setFeedbackStatus(prev => ({ ...prev, [movieId]: feedbackType }));
        api.sendFeedback(userId, movieId, feedbackType).catch(console.error);
        setTimeout(() => {
            setFeedbackStatus(prev => {
                const newState = {...prev};
                delete newState[movieId];
                return newState;
            });
        }, 2000);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
            <style>{`
                .scrollbar-hide::-webkit-scrollbar { display: none; }
                .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
            `}</style>
            <header className="bg-black/40 backdrop-blur-lg border-b border-purple-500/20 sticky top-0 z-50 shadow-xl">
                 <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
                     <div className="flex items-center space-x-3">
                         <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2.5 rounded-xl shadow-lg"><Film className="w-8 h-8 text-white" /></div>
                         <div><h1 className="text-2xl font-bold">MoviePulse</h1><p className="text-purple-300 text-sm">AI-Powered Movie Companion</p></div>
                     </div>
                     <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
                         <div className="flex items-center bg-white/10 border border-purple-500/30 rounded-lg p-1">
                             <button onClick={() => setView('homepage')} className={`px-3 py-1.5 rounded-md text-sm font-semibold flex items-center space-x-2 transition-colors ${view === 'homepage' ? 'bg-purple-600 text-white' : 'text-purple-300 hover:bg-white/10'}`}><Home className="w-4 h-4"/><span>Homepage</span></button>
                             <button onClick={() => setView('explorer')} className={`px-3 py-1.5 rounded-md text-sm font-semibold flex items-center space-x-2 transition-colors ${view === 'explorer' ? 'bg-purple-600 text-white' : 'text-purple-300 hover:bg-white/10'}`}><LayoutGrid className="w-4 h-4"/><span>Explorer</span></button>
                         </div>
                         <div className="flex items-center space-x-2 w-full sm:w-auto">
                            <div className="relative flex-grow"><input type="number" min="1" max="943" value={searchUserId} onChange={(e) => setSearchUserId(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSearch()} placeholder="Enter User ID" className="bg-white/10 border border-purple-500/30 rounded-lg px-4 py-2.5 pl-10 text-white w-full" /><User className="w-4 h-4 text-purple-300 absolute left-3 top-1/2 -translate-y-1/2" /></div>
                            <button onClick={handleSearch} className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-5 py-2.5 rounded-lg font-semibold flex items-center space-x-2"><Search className="w-4 h-4" /><span>Go</span></button>
                         </div>
                     </div>
                 </div>
            </header>
            
            {view === 'homepage' ? (
                <HomepageView loading={loading} error={error} data={homepageData} onSelectMovie={setSelectedMovie} />
            ) : (
                <ModelExplorerView userId={userId} loading={loading} error={error} recommendations={explorerRecs} onFeedback={handleFeedback} onSelectMovie={setSelectedMovie} activeTab={activeTab} onTabChange={handleTabChange} onRefresh={() => handleFetch(userId, activeTab)} feedbackStatus={feedbackStatus}/>
            )}
            
            <MovieModal movie={selectedMovie} onClose={() => setSelectedMovie(null)} onFeedback={handleFeedback}/>
        </div>
    );
}


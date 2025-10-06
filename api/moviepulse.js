const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Fetches recommendations from a specified model endpoint.
 * @param {string} userId The ID of the user.
 * @param {string} modelEndpoint The endpoint for the model (e.g., 'hybrid', 'user').
 * @returns {Promise<Array>} A promise that resolves to the list of recommendations.
 */
export const fetchRecommendations = async (userId, modelEndpoint) => {
    const response = await fetch(`${API_BASE_URL}/recommend/${modelEndpoint}/${userId}?count=8`);
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error fetching recommendations: ${response.status}`);
    }
    const data = await response.json();
    return data.recommendations || [];
};

/**
 * Sends user feedback to the API.
 * @param {string} userId The ID of the user.
 * @param {number} movieId The ID of the movie.
 * @param {'like' | 'dislike'} feedbackType The type of feedback.
 * @returns {Promise<Object>} A promise that resolves to the API response.
 */
export const sendFeedback = async (userId, movieId, feedbackType) => {
    const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: parseInt(userId, 10),
            movie_id: movieId,
            feedback_type: feedbackType,
        }),
    });
    if (!response.ok) {
        throw new Error('Feedback API request failed');
    }
    return response.json();
};


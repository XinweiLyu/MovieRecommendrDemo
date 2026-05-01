import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    RECOMMENDATION_DIR,
    RATING_MIN,
    RATING_MAX,
)

sns.set_theme(style="whitegrid")


def clip_rating(x):
    return float(np.clip(x, RATING_MIN, RATING_MAX))


class SVDRecommender:
    """
    This class is repeated here so joblib can load svd_recommender.joblib.
    The model was saved when 03_modeling.py ran as __main__, so Python expects
    to find SVDRecommender in the current __main__ script during loading.
    """

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.global_mean = None
        self.user_means = None
        self.train_matrix = None
        self.svd = None
        self.user_factors = None
        self.movie_factors = None
        self.user_to_idx = None
        self.movie_to_idx = None

    def predict_one(self, user_id, movie_id):
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean

        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        pred_centered = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        pred = self.user_means.loc[user_id] + pred_centered
        return clip_rating(pred)

    def predict(self, data):
        return np.array([
            self.predict_one(row.userId, row.movieId)
            for row in data.itertuples(index=False)
        ])


def recommend_movies_for_user(user_id, model, movies, ratings, top_n=10):
    all_movie_ids = movies["movieId"].unique()
    rated_movies = set(ratings.loc[ratings["userId"] == user_id, "movieId"].unique())
    candidate_movies = [m for m in all_movie_ids if m not in rated_movies]

    predictions = []
    for movie_id in candidate_movies:
        pred_rating = model.predict_one(user_id, movie_id)
        predictions.append((movie_id, pred_rating))

    recommendations = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
    recommendations = recommendations.merge(movies, on="movieId", how="left")
    recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(top_n)
    return recommendations


def plot_recommendations(recommendations, user_id):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=recommendations, y="title", x="predicted_rating")
    plt.title(f"Top Recommended Movies for User {user_id}")
    plt.xlabel("Predicted rating")
    plt.ylabel("Movie title")
    plt.xlim(0.5, 5.0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"15_top_recommendations_user_{user_id}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    user_id = 1
    top_n = 10

    movies = pd.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
    ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_clean.csv")

    # Use SVD matrix factorization for the final recommendation demo.
    model = load(MODEL_DIR / "svd_recommender.joblib")

    recommendations = recommend_movies_for_user(user_id, model, movies, ratings, top_n=top_n)
    output_path = RECOMMENDATION_DIR / f"top_{top_n}_recommendations_user_{user_id}.csv"
    recommendations.to_csv(output_path, index=False)

    plot_recommendations(recommendations, user_id)

    print(recommendations[["movieId", "title", "genres", "predicted_rating"]])
    print(f"Recommendations saved to {output_path}")


if __name__ == "__main__":
    main()

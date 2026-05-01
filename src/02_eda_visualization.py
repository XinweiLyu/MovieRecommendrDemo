import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DATA_DIR, FIGURE_DIR

sns.set_theme(style="whitegrid")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_clean.csv")
    movies = pd.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
    rating_movie = pd.read_csv(PROCESSED_DATA_DIR / "rating_movie_clean.csv")
    genre_data = pd.read_csv(PROCESSED_DATA_DIR / "genre_data_clean.csv")
    tags = pd.read_csv(PROCESSED_DATA_DIR / "tags_clean.csv")

    summary = pd.DataFrame({
        "metric": [
            "Number of users",
            "Number of movies with ratings",
            "Number of movies in metadata",
            "Number of ratings",
            "Number of tags",
            "Average rating",
            "Median rating",
            "Rating sparsity"
        ],
        "value": [
            ratings["userId"].nunique(),
            ratings["movieId"].nunique(),
            movies["movieId"].nunique(),
            len(ratings),
            len(tags),
            round(ratings["rating"].mean(), 4),
            round(ratings["rating"].median(), 4),
            round(1 - len(ratings) / (ratings["userId"].nunique() * movies["movieId"].nunique()), 4)
        ]
    })
    summary.to_csv(PROCESSED_DATA_DIR / "eda_summary.csv", index=False)
    print(summary)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=ratings, x="rating")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_fig("01_rating_distribution.png")

    user_counts = ratings.groupby("userId").size().reset_index(name="num_ratings")
    plt.figure(figsize=(8, 5))
    sns.histplot(user_counts["num_ratings"], bins=30, kde=True)
    plt.title("Distribution of Ratings per User")
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Number of users")
    save_fig("02_user_activity_distribution.png")

    movie_counts = ratings.groupby("movieId").size().reset_index(name="num_ratings")
    plt.figure(figsize=(8, 5))
    sns.histplot(movie_counts["num_ratings"], bins=50, kde=True)
    plt.title("Distribution of Ratings per Movie")
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Number of movies")
    save_fig("03_movie_popularity_distribution.png")

    top_movies = (
        rating_movie.groupby("title")
        .agg(num_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .sort_values("num_ratings", ascending=False)
        .head(10)
        .reset_index()
    )
    top_movies.to_csv(PROCESSED_DATA_DIR / "top_10_most_rated_movies.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_movies, y="title", x="num_ratings")
    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of ratings")
    plt.ylabel("Movie title")
    save_fig("04_top_10_most_rated_movies.png")

    genre_summary = (
        genre_data.groupby("genre")
        .agg(num_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .sort_values("num_ratings", ascending=False)
        .reset_index()
    )
    genre_summary.to_csv(PROCESSED_DATA_DIR / "genre_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=genre_summary, y="genre", x="num_ratings")
    plt.title("Number of Ratings by Genre")
    plt.xlabel("Number of ratings")
    plt.ylabel("Genre")
    save_fig("05_ratings_by_genre.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=genre_summary.sort_values("avg_rating", ascending=False), y="genre", x="avg_rating")
    plt.title("Average Rating by Genre")
    plt.xlabel("Average rating")
    plt.ylabel("Genre")
    save_fig("06_average_rating_by_genre.png")

    yearly_ratings = ratings.groupby("rating_year").size().reset_index(name="num_ratings")
    yearly_ratings.to_csv(PROCESSED_DATA_DIR / "ratings_by_year.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly_ratings, x="rating_year", y="num_ratings", marker="o")
    plt.title("Number of Ratings Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of ratings")
    save_fig("07_ratings_over_time.png")

    movie_year_summary = (
        rating_movie.dropna(subset=["movie_year"])
        .groupby("movie_year")
        .agg(num_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    movie_year_summary.to_csv(PROCESSED_DATA_DIR / "movie_year_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=movie_year_summary, x="movie_year", y="avg_rating")
    plt.title("Average Rating by Movie Release Year")
    plt.xlabel("Movie release year")
    plt.ylabel("Average rating")
    save_fig("08_average_rating_by_release_year.png")

    top_tags = tags["tag_clean"].value_counts().head(20).reset_index()
    top_tags.columns = ["tag", "count"]
    top_tags.to_csv(PROCESSED_DATA_DIR / "top_20_tags.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_tags, y="tag", x="count")
    plt.title("Top 20 User-generated Tags")
    plt.xlabel("Count")
    plt.ylabel("Tag")
    save_fig("09_top_20_tags.png")

    # Small sample heatmap only; the full user-item matrix is too sparse and large for a readable figure.
    user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    sampled_matrix = user_item.iloc[:50, :50]
    plt.figure(figsize=(10, 8))
    sns.heatmap(sampled_matrix, cmap="viridis", cbar_kws={"label": "Rating"})
    plt.title("Sample of User-Item Rating Matrix")
    plt.xlabel("Movie ID")
    plt.ylabel("User ID")
    save_fig("10_user_item_matrix_sample_heatmap.png")

    print(f"EDA completed. Figures are saved in {FIGURE_DIR}.")


if __name__ == "__main__":
    main()

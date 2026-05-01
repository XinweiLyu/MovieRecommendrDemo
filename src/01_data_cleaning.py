import re
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RATING_MIN, RATING_MAX


def load_csv_files():
    ratings = pd.read_csv(RAW_DATA_DIR / "ratings.csv", encoding="utf-8")
    movies = pd.read_csv(RAW_DATA_DIR / "movies.csv", encoding="utf-8")
    tags = pd.read_csv(RAW_DATA_DIR / "tags.csv", encoding="utf-8")
    links = pd.read_csv(RAW_DATA_DIR / "links.csv", encoding="utf-8")
    return ratings, movies, tags, links


def standardize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_required_columns(df, required_cols, file_name):
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def clean_ratings(ratings):
    ratings = standardize_columns(ratings)
    validate_required_columns(ratings, ["userId", "movieId", "rating", "timestamp"], "ratings.csv")

    ratings = ratings.drop_duplicates().copy()
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").astype("Int64")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["timestamp"] = ratings["timestamp"].astype(int)

    # Keep valid MovieLens ratings only: 0.5 to 5.0, in half-star increments.
    ratings = ratings[(ratings["rating"] >= RATING_MIN) & (ratings["rating"] <= RATING_MAX)]
    ratings = ratings[np.isclose((ratings["rating"] * 2) % 1, 0)]

    ratings["rating_datetime"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["rating_year"] = ratings["rating_datetime"].dt.year
    ratings["rating_month"] = ratings["rating_datetime"].dt.month

    return ratings


def extract_movie_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else np.nan


def remove_year_from_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def clean_movies(movies):
    movies = standardize_columns(movies)
    validate_required_columns(movies, ["movieId", "title", "genres"], "movies.csv")

    movies = movies.drop_duplicates().copy()
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")
    movies = movies.dropna(subset=["movieId", "title", "genres"])
    movies["movieId"] = movies["movieId"].astype(int)

    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].astype(str).str.strip()
    movies["movie_year"] = movies["title"].apply(extract_movie_year)
    movies["clean_title"] = movies["title"].apply(remove_year_from_title)
    movies["genres_clean"] = movies["genres"].replace("(no genres listed)", np.nan)
    movies["genre_list"] = movies["genres_clean"].fillna("Unknown").str.split("|")

    return movies


def clean_tags(tags):
    tags = standardize_columns(tags)
    validate_required_columns(tags, ["userId", "movieId", "tag", "timestamp"], "tags.csv")

    tags = tags.drop_duplicates().copy()
    tags["userId"] = pd.to_numeric(tags["userId"], errors="coerce").astype("Int64")
    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce").astype("Int64")
    tags["timestamp"] = pd.to_numeric(tags["timestamp"], errors="coerce").astype("Int64")
    tags["tag"] = tags["tag"].astype(str).str.strip()

    tags = tags.dropna(subset=["userId", "movieId", "tag", "timestamp"])
    tags = tags[tags["tag"] != ""]
    tags["userId"] = tags["userId"].astype(int)
    tags["movieId"] = tags["movieId"].astype(int)
    tags["timestamp"] = tags["timestamp"].astype(int)
    tags["tag_clean"] = tags["tag"].str.lower().str.replace(r"\s+", " ", regex=True)
    tags["tag_datetime"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)
    tags["tag_year"] = tags["tag_datetime"].dt.year

    return tags


def clean_links(links):
    links = standardize_columns(links)
    validate_required_columns(links, ["movieId", "imdbId", "tmdbId"], "links.csv")

    links = links.drop_duplicates().copy()
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["movieId", "imdbId"])
    links["movieId"] = links["movieId"].astype(int)
    links["imdbId"] = links["imdbId"].astype(int)

    # tmdbId has missing values in this dataset, so keep it as nullable Int64.
    return links


def build_analysis_tables(ratings, movies, tags, links):
    rating_movie = ratings.merge(movies, on="movieId", how="left")
    rating_movie = rating_movie.merge(links, on="movieId", how="left")

    genre_data = rating_movie.copy()
    genre_data["genre"] = genre_data["genre_list"].apply(lambda x: x if isinstance(x, list) else ["Unknown"])
    genre_data = genre_data.explode("genre")

    tag_movie = tags.merge(movies[["movieId", "title", "genres", "movie_year"]], on="movieId", how="left")

    return rating_movie, genre_data, tag_movie


def save_cleaned_data(ratings, movies, tags, links, rating_movie, genre_data, tag_movie):
    ratings.to_csv(PROCESSED_DATA_DIR / "ratings_clean.csv", index=False)
    movies.to_csv(PROCESSED_DATA_DIR / "movies_clean.csv", index=False)
    tags.to_csv(PROCESSED_DATA_DIR / "tags_clean.csv", index=False)
    links.to_csv(PROCESSED_DATA_DIR / "links_clean.csv", index=False)
    rating_movie.to_csv(PROCESSED_DATA_DIR / "rating_movie_clean.csv", index=False)
    genre_data.to_csv(PROCESSED_DATA_DIR / "genre_data_clean.csv", index=False)
    tag_movie.to_csv(PROCESSED_DATA_DIR / "tag_movie_clean.csv", index=False)

    user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    user_item_matrix.to_csv(PROCESSED_DATA_DIR / "user_item_matrix.csv")


def print_data_report(ratings, movies, tags, links):
    report = {
        "ratings_rows": len(ratings),
        "movies_rows": len(movies),
        "tags_rows": len(tags),
        "links_rows": len(links),
        "num_users": ratings["userId"].nunique(),
        "num_rated_movies": ratings["movieId"].nunique(),
        "num_movies_metadata": movies["movieId"].nunique(),
        "missing_tmdbId": int(links["tmdbId"].isna().sum()),
        "missing_movie_year": int(movies["movie_year"].isna().sum()),
        "global_average_rating": round(float(ratings["rating"].mean()), 4),
    }
    report_df = pd.DataFrame(list(report.items()), columns=["metric", "value"])
    report_df.to_csv(PROCESSED_DATA_DIR / "data_cleaning_report.csv", index=False)
    print(report_df)


def main():
    ratings_raw, movies_raw, tags_raw, links_raw = load_csv_files()

    ratings = clean_ratings(ratings_raw)
    movies = clean_movies(movies_raw)
    tags = clean_tags(tags_raw)
    links = clean_links(links_raw)

    rating_movie, genre_data, tag_movie = build_analysis_tables(ratings, movies, tags, links)
    save_cleaned_data(ratings, movies, tags, links, rating_movie, genre_data, tag_movie)
    print_data_report(ratings, movies, tags, links)

    print("\nCleaning completed. Cleaned files are saved in data/processed/.")


if __name__ == "__main__":
    main()

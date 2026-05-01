from __future__ import annotations

import __main__
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from joblib import load

from config import MODEL_DIR, PROCESSED_DATA_DIR, RATING_MAX, RATING_MIN


def clip_rating(x: float) -> float:
    return float(np.clip(x, RATING_MIN, RATING_MAX))


class SVDRecommender:
    """
    Keep the same class name for joblib deserialization compatibility.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.global_mean = None
        self.user_means = None
        self.train_matrix = None
        self.svd = None
        self.user_factors = None
        self.movie_factors = None
        self.user_to_idx = None
        self.movie_to_idx = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return float(self.global_mean)

        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        pred_centered = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        pred = self.user_means.loc[user_id] + pred_centered
        return clip_rating(pred)


class UserBasedCF:
    def __init__(self, k: int = 20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.user_similarity_df = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.train_matrix.index or movie_id not in self.train_matrix.columns:
            return float(self.global_mean)

        sim_scores = self.user_similarity_df[user_id].drop(labels=[user_id], errors="ignore")
        movie_ratings = self.train_matrix[movie_id]
        valid_users = movie_ratings.dropna().index

        if len(valid_users) == 0:
            return float(self.global_mean)

        sim_scores = sim_scores.loc[sim_scores.index.intersection(valid_users)]
        movie_ratings = movie_ratings.loc[sim_scores.index]

        if len(sim_scores) == 0:
            return float(self.global_mean)

        top_users = sim_scores.sort_values(ascending=False).head(self.k).index
        top_sim = sim_scores.loc[top_users]
        top_ratings = movie_ratings.loc[top_users]

        denom = np.abs(top_sim).sum()
        if denom == 0:
            return float(self.global_mean)

        return clip_rating(np.dot(top_sim, top_ratings) / denom)


class ItemBasedCF:
    def __init__(self, k: int = 20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.item_similarity_df = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.train_matrix.index or movie_id not in self.item_similarity_df.index:
            return float(self.global_mean)

        user_ratings = self.train_matrix.loc[user_id].dropna()
        if len(user_ratings) == 0:
            return float(self.global_mean)

        rated_movies = user_ratings.index.intersection(self.item_similarity_df.columns)
        sim_scores = self.item_similarity_df.loc[movie_id, rated_movies]
        sim_scores = sim_scores.drop(labels=[movie_id], errors="ignore")

        if len(sim_scores) == 0:
            return float(self.global_mean)

        top_items = sim_scores.sort_values(ascending=False).head(self.k).index
        top_sim = sim_scores.loc[top_items]
        top_ratings = user_ratings.loc[top_items]

        denom = np.abs(top_sim).sum()
        if denom == 0:
            return float(self.global_mean)

        return clip_rating(np.dot(top_sim, top_ratings) / denom)


class PopularityBaseline:
    def __init__(self):
        self.global_mean = None
        self.movie_mean = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        return clip_rating(self.movie_mean.get(movie_id, self.global_mean))


setattr(__main__, "SVDRecommender", SVDRecommender)
setattr(__main__, "UserBasedCF", UserBasedCF)
setattr(__main__, "ItemBasedCF", ItemBasedCF)
setattr(__main__, "PopularityBaseline", PopularityBaseline)


MODEL_FILES: dict[str, str] = {
    "svd": "svd_recommender.joblib",
    "user_cf": "user_based_cf.joblib",
    "item_cf": "item_based_cf.joblib",
    "popularity": "popularity_baseline.joblib",
}


@dataclass
class RecommendationService:
    models: dict[str, Any] | None = None
    movies: pd.DataFrame | None = None
    ratings: pd.DataFrame | None = None
    init_error: str | None = None

    @property
    def available_models(self) -> tuple[str, ...]:
        if not self.models:
            return ()
        return tuple(self.models.keys())

    def recommend(self, user_id: int, top_n: int, model_name: str) -> list[dict[str, Any]]:
        if self.models is None or self.movies is None or self.ratings is None:
            if self.init_error:
                raise RuntimeError(self.init_error)
            raise RuntimeError("Recommendation service is not initialized.")

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Available models: {', '.join(self.available_models)}"
            )

        all_movie_ids = self.movies["movieId"].unique()
        rated_movies = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"].unique())
        candidate_movies = [int(movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]

        predictions = []
        for movie_id in candidate_movies:
            pred_rating = float(model.predict_one(user_id, movie_id))
            predictions.append((movie_id, pred_rating))

        recommendations = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
        recommendations = recommendations.merge(self.movies, on="movieId", how="left")
        recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(top_n)

        response_fields = [col for col in ("movieId", "title", "genres") if col in recommendations]
        return recommendations[response_fields].to_dict(orient="records")


service = RecommendationService()
service_lock = threading.Lock()


def init_recommendation_service() -> None:
    with service_lock:
        if (
            getattr(service, "models", None) is not None
            and getattr(service, "movies", None) is not None
            and getattr(service, "ratings", None) is not None
        ):
            return

        try:
            service.movies = pd.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
            service.ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_clean.csv")
            loaded_models: dict[str, Any] = {}
            for model_name, model_file in MODEL_FILES.items():
                model_path = MODEL_DIR / model_file
                if model_path.exists():
                    loaded_models[model_name] = load(model_path)
            if not loaded_models:
                raise FileNotFoundError(
                    f"No model file found under {MODEL_DIR}. Expected one of: {', '.join(MODEL_FILES.values())}"
                )
            service.models = loaded_models
            service.init_error = None
        except Exception as exc:
            service.models = None
            service.movies = None
            service.ratings = None
            service.init_error = (
                "Failed to initialize recommendation service. "
                "Please run the training pipeline first and ensure model/data files exist. "
                f"Root cause: {exc}"
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_recommendation_service()
    yield


app = FastAPI(title="Movie Recommender MVP API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/recommendations")
def get_recommendations(
    user_id: int = Query(..., gt=0),
    top_n: int = Query(10, ge=1, le=50),
    model_name: str = Query("svd"),
) -> dict[str, Any]:
    init_recommendation_service()
    try:
        if model_name not in service.available_models:
            available_str = ", ".join(service.available_models)
            raise HTTPException(status_code=422, detail=f"model_name must be one of: {available_str}")
        recommendations = service.recommend(user_id=user_id, top_n=top_n, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "user_id": user_id,
        "top_n": top_n,
        "model_name": model_name,
        "available_models": service.available_models,
        "recommendations": recommendations,
    }


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mvp_api:app", host="0.0.0.0", port=8000, reload=False)

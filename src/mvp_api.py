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

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        if user_id not in self.user_to_idx:
            return np.full(len(movie_ids), float(self.global_mean), dtype=float)

        user_idx = self.user_to_idx[user_id]
        user_factor = self.user_factors[user_idx]
        user_mean = float(self.user_means.loc[user_id])
        predictions = np.full(len(movie_ids), float(self.global_mean), dtype=float)
        known_mask = np.array([movie_id in self.movie_to_idx for movie_id in movie_ids], dtype=bool)
        if not np.any(known_mask):
            return predictions

        known_movie_ids = [movie_ids[idx] for idx, known in enumerate(known_mask) if known]
        movie_indices = np.array([self.movie_to_idx[movie_id] for movie_id in known_movie_ids], dtype=int)
        pred_centered = self.movie_factors[movie_indices] @ user_factor
        predictions[known_mask] = np.clip(user_mean + pred_centered, RATING_MIN, RATING_MAX)
        return predictions


class UserBasedCF:
    def __init__(self, k: int = 20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.user_similarity_df = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        return float(self.predict_many(user_id, [movie_id])[0])

    def _ensure_fast_cache(self) -> None:
        if getattr(self, "_fast_cache_ready", False):
            return
        self._user_ids = np.array(self.train_matrix.index, dtype=int)
        self._movie_ids = np.array(self.train_matrix.columns, dtype=int)
        self._user_to_pos = {int(user_id): idx for idx, user_id in enumerate(self._user_ids)}
        self._movie_to_pos = {int(movie_id): idx for idx, movie_id in enumerate(self._movie_ids)}
        self._train_matrix_values = self.train_matrix.to_numpy(dtype=np.float32, copy=False)
        self._user_similarity_values = self.user_similarity_df.to_numpy(dtype=np.float32, copy=False)
        self._fast_cache_ready = True

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        self._ensure_fast_cache()
        predictions = np.full(len(movie_ids), float(self.global_mean), dtype=float)

        user_pos = self._user_to_pos.get(int(user_id))
        if user_pos is None:
            return predictions

        candidate_positions = np.array([self._movie_to_pos.get(int(movie_id), -1) for movie_id in movie_ids], dtype=int)
        known_mask = candidate_positions >= 0
        if not np.any(known_mask):
            return predictions

        known_candidate_positions = candidate_positions[known_mask]
        ratings_sub = self._train_matrix_values[:, known_candidate_positions]  # (num_users, num_candidates)
        valid_mask = ~np.isnan(ratings_sub)
        if not np.any(valid_mask):
            return predictions

        sim_scores = self._user_similarity_values[user_pos].astype(np.float32, copy=True)
        sim_scores[user_pos] = -np.inf  # exclude self-neighbor

        # Candidate-wise user similarities; invalid raters are masked out.
        sim_matrix = np.where(valid_mask.T, sim_scores[None, :], -np.inf)  # (num_candidates, num_users)
        ratings_matrix = ratings_sub.T.astype(np.float32, copy=False)  # (num_candidates, num_users)

        k = min(self.k, sim_matrix.shape[1])
        if k <= 0:
            return predictions

        if k < sim_matrix.shape[1]:
            top_idx = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k]
            top_sim = np.take_along_axis(sim_matrix, top_idx, axis=1)
            top_ratings = np.take_along_axis(ratings_matrix, top_idx, axis=1)
        else:
            top_sim = sim_matrix
            top_ratings = ratings_matrix

        invalid = ~np.isfinite(top_sim)
        top_sim = np.where(invalid, 0.0, top_sim)
        top_ratings = np.where(invalid, 0.0, top_ratings)

        denom = np.abs(top_sim).sum(axis=1)
        weighted_sum = (top_sim * top_ratings).sum(axis=1)
        known_predictions = np.full(len(known_candidate_positions), float(self.global_mean), dtype=float)
        np.divide(weighted_sum, denom, out=known_predictions, where=denom != 0)
        predictions[known_mask] = np.clip(known_predictions, RATING_MIN, RATING_MAX)
        return predictions


class ItemBasedCF:
    def __init__(self, k: int = 20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.item_similarity_df = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        return float(self.predict_many(user_id, [movie_id])[0])

    def _ensure_fast_cache(self) -> None:
        if getattr(self, "_fast_cache_ready", False):
            return
        self._user_ids = np.array(self.train_matrix.index, dtype=int)
        self._movie_ids = np.array(self.train_matrix.columns, dtype=int)
        self._user_to_pos = {int(user_id): idx for idx, user_id in enumerate(self._user_ids)}
        self._movie_to_pos = {int(movie_id): idx for idx, movie_id in enumerate(self._movie_ids)}
        self._train_matrix_values = self.train_matrix.to_numpy(dtype=np.float32, copy=False)
        self._item_similarity_values = self.item_similarity_df.to_numpy(dtype=np.float32, copy=False)
        self._fast_cache_ready = True

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        self._ensure_fast_cache()
        predictions = np.full(len(movie_ids), float(self.global_mean), dtype=float)

        user_pos = self._user_to_pos.get(int(user_id))
        if user_pos is None:
            return predictions

        user_row = self._train_matrix_values[user_pos]
        rated_positions = np.flatnonzero(~np.isnan(user_row))
        if rated_positions.size == 0:
            return predictions

        rated_values = user_row[rated_positions]
        candidate_positions = np.array([self._movie_to_pos.get(int(movie_id), -1) for movie_id in movie_ids], dtype=int)
        known_mask = candidate_positions >= 0
        if not np.any(known_mask):
            return predictions

        known_candidate_positions = candidate_positions[known_mask]
        similarity_matrix = self._item_similarity_values[known_candidate_positions[:, None], rated_positions[None, :]]
        if similarity_matrix.size == 0:
            return predictions

        k = min(self.k, similarity_matrix.shape[1])
        if k <= 0:
            return predictions

        if k < similarity_matrix.shape[1]:
            top_idx = np.argpartition(-similarity_matrix, kth=k - 1, axis=1)[:, :k]
            top_sim = np.take_along_axis(similarity_matrix, top_idx, axis=1)
            top_ratings = rated_values[top_idx]
        else:
            top_sim = similarity_matrix
            top_ratings = np.broadcast_to(rated_values, top_sim.shape)

        denom = np.abs(top_sim).sum(axis=1)
        weighted_sum = (top_sim * top_ratings).sum(axis=1)
        known_predictions = np.full(len(known_candidate_positions), float(self.global_mean), dtype=float)
        np.divide(weighted_sum, denom, out=known_predictions, where=denom != 0)
        predictions[known_mask] = np.clip(known_predictions, RATING_MIN, RATING_MAX)
        return predictions


class PopularityBaseline:
    def __init__(self):
        self.global_mean = None
        self.movie_mean = None

    def predict_one(self, user_id: int, movie_id: int) -> float:
        return clip_rating(self.movie_mean.get(movie_id, self.global_mean))

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        return np.array([clip_rating(self.movie_mean.get(movie_id, self.global_mean)) for movie_id in movie_ids], dtype=float)


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


class EmbeddingRecommenderAdapter:
    def __init__(self, model: Any, user_encoder: Any, movie_encoder: Any, fallback: float):
        self.model = model
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.fallback = fallback

    def predict_one(self, user_id: int, movie_id: int) -> float:
        # Unknown IDs are possible for cold-start requests.
        if user_id not in self.user_encoder.classes_ or movie_id not in self.movie_encoder.classes_:
            return float(self.fallback)

        user_idx = int(self.user_encoder.transform([user_id])[0])
        movie_idx = int(self.movie_encoder.transform([movie_id])[0])
        pred = self.model.predict([np.array([user_idx]), np.array([movie_idx])], verbose=0).flatten()[0]
        return clip_rating(float(pred))

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        if user_id not in self.user_encoder.classes_:
            return np.full(len(movie_ids), float(self.fallback), dtype=float)

        user_idx = int(self.user_encoder.transform([user_id])[0])
        movie_id_arr = np.array(movie_ids, dtype=int)
        known_mask = np.isin(movie_id_arr, self.movie_encoder.classes_)
        predictions = np.full(len(movie_ids), float(self.fallback), dtype=float)
        if not np.any(known_mask):
            return predictions

        known_movie_ids = movie_id_arr[known_mask]
        movie_indices = self.movie_encoder.transform(known_movie_ids).astype(int)
        user_indices = np.full(len(known_movie_ids), user_idx, dtype=int)
        known_preds = self.model.predict([user_indices, movie_indices], verbose=0).flatten()
        predictions[known_mask] = np.clip(known_preds, RATING_MIN, RATING_MAX)
        return predictions


def try_load_embedding_model(fallback: float) -> tuple[Any | None, str | None]:
    model_path = MODEL_DIR / "embedding_recommender.keras"
    user_encoder_path = MODEL_DIR / "user_encoder.joblib"
    movie_encoder_path = MODEL_DIR / "movie_encoder.joblib"
    if not (model_path.exists() and user_encoder_path.exists() and movie_encoder_path.exists()):
        return None, (
            "Missing embedding artifacts. Required files: "
            "embedding_recommender.keras, user_encoder.joblib, movie_encoder.joblib"
        )

    try:
        import tensorflow as tf
    except Exception as exc:
        return None, f"Failed to import TensorFlow: {exc}"

    try:
        model = tf.keras.models.load_model(model_path)
        user_encoder = load(user_encoder_path)
        movie_encoder = load(movie_encoder_path)
        return (
            EmbeddingRecommenderAdapter(
                model=model,
                user_encoder=user_encoder,
                movie_encoder=movie_encoder,
                fallback=fallback,
            ),
            None,
        )
    except Exception as exc:
        return None, f"Failed to load embedding model artifacts: {exc}"


@dataclass
class RecommendationService:
    models: dict[str, Any] | None = None
    movies: pd.DataFrame | None = None
    ratings: pd.DataFrame | None = None
    init_error: str | None = None
    model_load_errors: dict[str, str] | None = None

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

        if hasattr(model, "predict_many"):
            pred_scores = model.predict_many(user_id, candidate_movies)
            predictions = list(zip(candidate_movies, pred_scores.tolist()))
        else:
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
            model_load_errors: dict[str, str] = {}
            for model_name, model_file in MODEL_FILES.items():
                model_path = MODEL_DIR / model_file
                if model_path.exists():
                    loaded_models[model_name] = load(model_path)

            embedding_model, embedding_error = try_load_embedding_model(fallback=float(service.ratings["rating"].mean()))
            if embedding_model is not None:
                loaded_models["embedding"] = embedding_model
            elif embedding_error is not None:
                model_load_errors["embedding"] = embedding_error

            if not loaded_models:
                raise FileNotFoundError(
                    f"No model file found under {MODEL_DIR}. Expected one of: {', '.join(MODEL_FILES.values())}"
                )
            service.models = loaded_models
            service.init_error = None
            service.model_load_errors = model_load_errors
        except Exception as exc:
            service.models = None
            service.movies = None
            service.ratings = None
            service.model_load_errors = None
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
            detail = f"model_name must be one of: {available_str}"
            if model_name == "embedding":
                embedding_reason = (service.model_load_errors or {}).get("embedding")
                if embedding_reason:
                    detail = f"{detail}. embedding unavailable: {embedding_reason}"
            raise HTTPException(status_code=422, detail=detail)
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

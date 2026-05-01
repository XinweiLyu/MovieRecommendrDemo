from pathlib import Path
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mvp_api


@pytest.fixture
def client_with_stub_service(monkeypatch):
    class StubService:
        init_error = None
        available_models = ("svd", "user_cf")

        def recommend(self, user_id, top_n, model_name):
            return [
                {"movieId": 1, "title": "Movie A", "genres": "Drama"},
                {"movieId": 2, "title": "Movie B", "genres": "Comedy"},
            ][:top_n]

    monkeypatch.setattr(mvp_api, "service", StubService())
    return TestClient(mvp_api.app)


def test_health_endpoint(client_with_stub_service):
    response = client_with_stub_service.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_endpoint_success(client_with_stub_service):
    response = client_with_stub_service.get(
        "/api/recommendations",
        params={"user_id": 1, "top_n": 2, "model_name": "user_cf"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 1
    assert payload["top_n"] == 2
    assert payload["model_name"] == "user_cf"
    assert len(payload["recommendations"]) == 2
    assert payload["recommendations"][0]["movieId"] == 1


def test_recommendations_endpoint_invalid_params(client_with_stub_service):
    response = client_with_stub_service.get("/api/recommendations", params={"user_id": 0, "top_n": 100})
    assert response.status_code == 422


def test_recommendations_endpoint_invalid_model_name(client_with_stub_service):
    response = client_with_stub_service.get(
        "/api/recommendations",
        params={"user_id": 1, "top_n": 10, "model_name": "unknown"},
    )
    assert response.status_code == 422


def test_recommend_service_filters_rated_movies():
    class StubModelA:
        def predict_one(self, user_id, movie_id):
            return 5.0 - movie_id * 0.1

    class StubModelB:
        def predict_one(self, user_id, movie_id):
            return 1.0 + movie_id * 0.1

    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Movie A"},
            {"movieId": 2, "title": "Movie B"},
            {"movieId": 3, "title": "Movie C"},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.0},
        ]
    )
    service = mvp_api.RecommendationService(
        models={"svd": StubModelA(), "user_cf": StubModelB()},
        movies=movies,
        ratings=ratings,
    )

    recs = service.recommend(user_id=1, top_n=2, model_name="svd")
    rec_movie_ids = [item["movieId"] for item in recs]
    assert 1 not in rec_movie_ids
    assert rec_movie_ids == [2, 3]

    recs_alt = service.recommend(user_id=1, top_n=2, model_name="user_cf")
    assert [item["movieId"] for item in recs_alt] == [3, 2]

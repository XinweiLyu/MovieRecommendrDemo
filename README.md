# MovieLens Recommendation System Project

This project builds a complete movie recommendation pipeline using the MovieLens `ml-latest-small` dataset.

## Folder Structure

```text
movie_recommender_project/
├── data/
│   ├── raw/
│   │   ├── links.csv
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── tags.csv
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── recommendations/
├── src/
│   ├── config.py
│   ├── 01_data_cleaning.py
│   ├── 02_eda_visualization.py
│   ├── 03_modeling.py
│   └── 04_recommendation_demo.py
├── requirements.txt
└── run_all.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the whole project:

```bash
python run_all.py
```

Or run each step separately:

```bash
python src/01_data_cleaning.py
python src/02_eda_visualization.py
python src/03_modeling.py
python src/04_recommendation_demo.py
```

## MVP Frontend-Backend Runbook

Run the MVP API from the project root (`movie_recommender_project_finished`):

```bash
# Option 1: run with uvicorn
uvicorn mvp_api:app --app-dir src --host 0.0.0.0 --port 8000

# Option 2: run the module directly
python src/mvp_api.py
```

Then open:

- Frontend page: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`

### MVP Acceptance Verification Steps

1. Verify service startup:

```bash
curl "http://127.0.0.1:8000/health"
```

Expected response:

```json
{"status":"ok"}
```

2. Verify recommendation endpoint:

```bash
curl "http://127.0.0.1:8000/api/recommendations?user_id=1&top_n=5"
```

Expected behavior:

- HTTP `200`
- JSON includes `user_id`, `top_n`, and `recommendations`
- Each recommendation contains at least `movieId`, `title`, and `predicted_rating`

3. Verify frontend interaction:

- Open `http://127.0.0.1:8000/`
- Input `userId` and `topN`
- Click the query button
- Confirm a recommendation table is rendered
- If request fails, confirm the page shows a readable error message without crashing

## What Each Script Does

### `01_data_cleaning.py`

- Loads `ratings.csv`, `movies.csv`, `tags.csv`, and `links.csv`.
- Standardizes column names.
- Checks required columns.
- Removes duplicate rows.
- Converts data types.
- Converts Unix timestamps to UTC datetime.
- Extracts movie release year from title.
- Cleans title without year.
- Splits genres into a genre list.
- Keeps missing `tmdbId` as nullable integer because `links.csv` has missing TMDB IDs.
- Saves cleaned data into `data/processed/`.

### `02_eda_visualization.py`

Creates data representation and visualization:

- Rating distribution.
- User activity distribution.
- Movie popularity distribution.
- Top 10 most rated movies.
- Number of ratings by genre.
- Average rating by genre.
- Ratings over time.
- Average rating by movie release year.
- Top 20 tags.
- Sample user-item matrix heatmap.

All figures are saved in `outputs/figures/`.

### `03_modeling.py`

Builds, trains, validates, and tests:

- Popularity baseline.
- User-based collaborative filtering.
- Item-based collaborative filtering.
- SVD matrix factorization.
- Neural embedding recommender, if TensorFlow is installed.

Model performance is evaluated using:

- RMSE.
- MAE.

Results are saved in `data/processed/model_results.csv` and visualized in `outputs/figures/`.

### `04_recommendation_demo.py`

Uses the trained SVD model to generate top movie recommendations for a sample user.

Outputs:

- Recommendation CSV in `outputs/recommendations/`.
- Recommendation bar chart in `outputs/figures/`.

## Notes

The collaborative filtering models use the user-item rating matrix. Missing ratings mean a user has not rated that movie. The SVD and embedding models learn latent user and movie representations, which helps with sparse rating data.

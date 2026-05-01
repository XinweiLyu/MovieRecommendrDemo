import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from config import PROCESSED_DATA_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_STATE, RATING_MIN, RATING_MAX

sns.set_theme(style="whitegrid")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def clip_rating(x):
    return float(np.clip(x, RATING_MIN, RATING_MAX))


def split_data(ratings):
    train_data, temp_data = train_test_split(ratings, test_size=0.30, random_state=RANDOM_STATE)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=RANDOM_STATE)
    return train_data.copy(), val_data.copy(), test_data.copy()


class UserBasedCF:
    def __init__(self, k=20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.user_similarity_df = None

    def fit(self, train_data):
        self.global_mean = float(train_data["rating"].mean())
        self.train_matrix = train_data.pivot_table(index="userId", columns="movieId", values="rating")
        matrix_filled = self.train_matrix.fillna(0)
        user_similarity = cosine_similarity(matrix_filled).astype(np.float32)
        self.user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.train_matrix.index,
            columns=self.train_matrix.index
        )
        return self

    def predict_one(self, user_id, movie_id):
        if user_id not in self.train_matrix.index or movie_id not in self.train_matrix.columns:
            return self.global_mean

        sim_scores = self.user_similarity_df[user_id].drop(labels=[user_id], errors="ignore")
        movie_ratings = self.train_matrix[movie_id]
        valid_users = movie_ratings.dropna().index

        if len(valid_users) == 0:
            return self.global_mean

        sim_scores = sim_scores.loc[sim_scores.index.intersection(valid_users)]
        movie_ratings = movie_ratings.loc[sim_scores.index]

        if len(sim_scores) == 0:
            return self.global_mean

        top_users = sim_scores.sort_values(ascending=False).head(self.k).index
        top_sim = sim_scores.loc[top_users]
        top_ratings = movie_ratings.loc[top_users]

        denom = np.abs(top_sim).sum()
        if denom == 0:
            return self.global_mean

        return clip_rating(np.dot(top_sim, top_ratings) / denom)

    def predict(self, data):
        return np.array([self.predict_one(row.userId, row.movieId) for row in data.itertuples(index=False)])


class ItemBasedCF:
    def __init__(self, k=20):
        self.k = k
        self.global_mean = None
        self.train_matrix = None
        self.item_similarity_df = None

    def fit(self, train_data):
        self.global_mean = float(train_data["rating"].mean())
        self.train_matrix = train_data.pivot_table(index="userId", columns="movieId", values="rating")
        item_matrix = self.train_matrix.T.fillna(0)
        item_similarity = cosine_similarity(item_matrix).astype(np.float32)
        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )
        return self

    def predict_one(self, user_id, movie_id):
        if user_id not in self.train_matrix.index or movie_id not in self.item_similarity_df.index:
            return self.global_mean

        user_ratings = self.train_matrix.loc[user_id].dropna()
        if len(user_ratings) == 0:
            return self.global_mean

        rated_movies = user_ratings.index.intersection(self.item_similarity_df.columns)
        sim_scores = self.item_similarity_df.loc[movie_id, rated_movies]
        sim_scores = sim_scores.drop(labels=[movie_id], errors="ignore")

        if len(sim_scores) == 0:
            return self.global_mean

        top_items = sim_scores.sort_values(ascending=False).head(self.k).index
        top_sim = sim_scores.loc[top_items]
        top_ratings = user_ratings.loc[top_items]

        denom = np.abs(top_sim).sum()
        if denom == 0:
            return self.global_mean

        return clip_rating(np.dot(top_sim, top_ratings) / denom)

    def predict(self, data):
        return np.array([self.predict_one(row.userId, row.movieId) for row in data.itertuples(index=False)])


class SVDRecommender:
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

    def fit(self, train_data):
        self.global_mean = float(train_data["rating"].mean())
        self.train_matrix = train_data.pivot_table(index="userId", columns="movieId", values="rating")
        self.user_means = self.train_matrix.mean(axis=1).fillna(self.global_mean)

        centered = self.train_matrix.sub(self.user_means, axis=0).fillna(0)
        n_components = min(self.n_components, min(centered.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        self.user_factors = self.svd.fit_transform(centered)
        self.movie_factors = self.svd.components_.T

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.train_matrix.index)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.train_matrix.columns)}
        return self

    def predict_one(self, user_id, movie_id):
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean

        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        pred_centered = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        pred = self.user_means.loc[user_id] + pred_centered
        return clip_rating(pred)

    def predict(self, data):
        return np.array([self.predict_one(row.userId, row.movieId) for row in data.itertuples(index=False)])


class PopularityBaseline:
    def fit(self, train_data):
        self.global_mean = float(train_data["rating"].mean())
        self.movie_mean = train_data.groupby("movieId")["rating"].mean().to_dict()
        return self

    def predict_one(self, user_id, movie_id):
        return clip_rating(self.movie_mean.get(movie_id, self.global_mean))

    def predict(self, data):
        return np.array([self.predict_one(row.userId, row.movieId) for row in data.itertuples(index=False)])


def evaluate_model(name, model, train_data, val_data, test_data, sample_for_cf=False):
    model.fit(train_data)

    val_eval = val_data
    test_eval = test_data
    if sample_for_cf:
        val_eval = val_data.sample(min(3000, len(val_data)), random_state=RANDOM_STATE)
        test_eval = test_data.sample(min(3000, len(test_data)), random_state=RANDOM_STATE)

    val_pred = model.predict(val_eval)
    test_pred = model.predict(test_eval)

    result = {
        "Model": name,
        "Validation_RMSE": rmse(val_eval["rating"], val_pred),
        "Validation_MAE": mae(val_eval["rating"], val_pred),
        "Test_RMSE": rmse(test_eval["rating"], test_pred),
        "Test_MAE": mae(test_eval["rating"], test_pred),
        "Validation_Size": len(val_eval),
        "Test_Size": len(test_eval)
    }
    return result, model


def train_embedding_model(ratings, train_data, val_data, test_data):
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        print("TensorFlow is not available. Skipping Embedding Recommender.")
        print("Install it with: pip install tensorflow")
        print("Error:", exc)
        return None, None, None

    all_data = ratings.copy()
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    all_data["user_idx"] = user_encoder.fit_transform(all_data["userId"])
    all_data["movie_idx"] = movie_encoder.fit_transform(all_data["movieId"])

    train_nn = all_data.loc[train_data.index]
    val_nn = all_data.loc[val_data.index]
    test_nn = all_data.loc[test_data.index]

    num_users = all_data["user_idx"].nunique()
    num_movies = all_data["movie_idx"].nunique()
    embedding_dim = 50

    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")

    user_embedding = Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    movie_embedding = Embedding(num_movies, embedding_dim, name="movie_embedding")(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    x = Concatenate()([user_vec, movie_vec])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(
        [train_nn["user_idx"], train_nn["movie_idx"]],
        train_nn["rating"],
        validation_data=([val_nn["user_idx"], val_nn["movie_idx"]], val_nn["rating"]),
        epochs=15,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    test_pred = model.predict([test_nn["user_idx"], test_nn["movie_idx"]], verbose=0).flatten()
    test_pred = np.clip(test_pred, RATING_MIN, RATING_MAX)
    val_pred = model.predict([val_nn["user_idx"], val_nn["movie_idx"]], verbose=0).flatten()
    val_pred = np.clip(val_pred, RATING_MIN, RATING_MAX)

    result = {
        "Model": "Embedding Recommender",
        "Validation_RMSE": rmse(val_nn["rating"], val_pred),
        "Validation_MAE": mae(val_nn["rating"], val_pred),
        "Test_RMSE": rmse(test_nn["rating"], test_pred),
        "Test_MAE": mae(test_nn["rating"], test_pred),
        "Validation_Size": len(val_nn),
        "Test_Size": len(test_nn)
    }

    model.save(MODEL_DIR / "embedding_recommender.keras")
    dump(user_encoder, MODEL_DIR / "user_encoder.joblib")
    dump(movie_encoder, MODEL_DIR / "movie_encoder.joblib")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Embedding Recommender Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "13_embedding_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Embedding Recommender Training and Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "14_embedding_training_mae.png", dpi=300, bbox_inches="tight")
    plt.close()

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(PROCESSED_DATA_DIR / "embedding_training_history.csv", index=False)

    return result, model, history_df


def plot_model_results(results_df):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Model", y="Test_RMSE")
    plt.title("Model Comparison by Test RMSE")
    plt.xticks(rotation=20)
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_model_comparison_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Model", y="Test_MAE")
    plt.title("Model Comparison by Test MAE")
    plt.xticks(rotation=20)
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "12_model_comparison_mae.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_clean.csv")
    ratings = ratings.reset_index(drop=True)
    train_data, val_data, test_data = split_data(ratings)

    train_data.to_csv(PROCESSED_DATA_DIR / "train_ratings.csv", index=False)
    val_data.to_csv(PROCESSED_DATA_DIR / "validation_ratings.csv", index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / "test_ratings.csv", index=False)

    results = []

    baseline_result, baseline_model = evaluate_model(
        "Popularity Baseline", PopularityBaseline(), train_data, val_data, test_data
    )
    results.append(baseline_result)
    dump(baseline_model, MODEL_DIR / "popularity_baseline.joblib")

    user_cf_result, user_cf_model = evaluate_model(
        "User-based CF", UserBasedCF(k=20), train_data, val_data, test_data, sample_for_cf=True
    )
    results.append(user_cf_result)
    dump(user_cf_model, MODEL_DIR / "user_based_cf.joblib")

    item_cf_result, item_cf_model = evaluate_model(
        "Item-based CF", ItemBasedCF(k=20), train_data, val_data, test_data, sample_for_cf=True
    )
    results.append(item_cf_result)
    dump(item_cf_model, MODEL_DIR / "item_based_cf.joblib")

    svd_result, svd_model = evaluate_model(
        "SVD Matrix Factorization", SVDRecommender(n_components=50), train_data, val_data, test_data
    )
    results.append(svd_result)
    dump(svd_model, MODEL_DIR / "svd_recommender.joblib")

    embedding_result, embedding_model, embedding_history = train_embedding_model(ratings, train_data, val_data, test_data)
    if embedding_result is not None:
        results.append(embedding_result)

    results_df = pd.DataFrame(results).sort_values("Test_RMSE")
    results_df.to_csv(PROCESSED_DATA_DIR / "model_results.csv", index=False)
    plot_model_results(results_df)

    with open(MODEL_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(results_df)
    print(f"Modeling completed. Results saved to {PROCESSED_DATA_DIR / 'model_results.csv'}")


if __name__ == "__main__":
    main()

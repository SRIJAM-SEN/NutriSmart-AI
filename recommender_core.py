"""
recommender_core.py

Core ML + data helper functions for NutriSmart AI food recommender.

- load_dataset(path)
- get_recommendations(...)

Expected CSV columns (case-insensitive / flexible):
Food_ID, Food_Name, Type, Calories, Protein, Fat, Carbs,
Meal_Type, Price, Food_Image_URL, Cuisine
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

# ------------- CONFIG -------------
TFIDF_MAX_FEATURES = 3000
TOP_K_DEFAULT = 12


# ------------- DATA LOADER -------------
def load_dataset(path) -> pd.DataFrame:
    """
    Load and clean the main food CSV.

    Uses encoding='latin1' to avoid UnicodeDecodeError.
    Creates canonical columns:
      - food_name, type, meal_type, image_url, cuisine
      - _calories_f, _protein_f, _fat_f, _carbs_f, _price_f
      - ingredients (proxy, from food_name)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")

    # IMPORTANT: encoding to fix UnicodeDecodeError
    df = pd.read_csv(p, encoding="latin1")

    # Normalize headers and strip BOM / whitespace
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns.astype(str)]

    # Canonical text columns (fall back if names slightly different)
    df["food_name"] = df.get("Food_Name", df.columns[0]).astype(str).fillna("").str.strip()
    df["type"] = df.get("Type", "").astype(str).str.strip()
    df["meal_type"] = df.get("Meal_Type", "").astype(str).str.strip()
    df["image_url"] = df.get("Food_Image_URL", "").astype(str).str.strip()
    df["cuisine"] = df.get("Cuisine", "").astype(str).str.strip()

    # Numeric columns with safety
    for col in ["Calories", "Protein", "Fat", "Carbs", "Price"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Canonical numeric copies
    df["_calories_f"] = df["Calories"].astype(float)
    df["_protein_f"] = df["Protein"].astype(float)
    df["_fat_f"] = df["Fat"].astype(float)
    df["_carbs_f"] = df["Carbs"].astype(float)
    df["_price_f"] = df["Price"].astype(float)

    # Ingredients proxy (you can later replace by real ingredients column)
    df["ingredients"] = df["food_name"].fillna("").astype(str).str.lower()

    # Remove duplicate food names (keep first)
    df = df.drop_duplicates(subset=["food_name"], keep="first").reset_index(drop=True)

    return df


# ------------- FILTERING -------------
def _filter_dataset(df: pd.DataFrame,
                    food_type: str,
                    meal_types,
                    price_min: float,
                    price_max: float,
                    dislike_words):
    out = df.copy()

    # Type filter
    if food_type and str(food_type).lower() != "any":
        out = out[out["type"].str.lower() == str(food_type).lower()]

    # Meal type filter
    if meal_types:
        out = out[out["meal_type"].isin(meal_types)]

    # Price range filter
    out = out[(out["_price_f"] >= price_min) & (out["_price_f"] <= price_max)]

    # Dislikes filter on food_name + ingredients
    if dislike_words:
        def ok(txt):
            s = " " + str(txt).lower() + " "
            for w in dislike_words:
                w = w.strip().lower()
                if not w:
                    continue
                if w in s:
                    return False
            return True

        mask = out["food_name"].apply(ok) & out["ingredients"].apply(ok)
        out = out[mask]

    return out


# ------------- ML: HYBRID MATRIX -------------
def _build_hybrid_matrix(df_filtered: pd.DataFrame,
                         tfidf_max_features: int = TFIDF_MAX_FEATURES):
    """
    Build hybrid feature matrix:
      - TF-IDF on (food_name + ingredients)
      - Scaled numeric features [calories, protein, fat, carbs, price]
    Returns:
      hybrid_matrix (CSR), vectorizer, scaler
    """
    # Text corpus
    corpus = (df_filtered["food_name"].fillna("") + " " +
              df_filtered["ingredients"].fillna("")).values

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_text = vectorizer.fit_transform(corpus)  # sparse

    # Numeric features
    num_cols = ["_calories_f", "_protein_f", "_fat_f", "_carbs_f", "_price_f"]
    numeric = df_filtered[num_cols].values
    scaler = StandardScaler()
    X_num = scaler.fit_transform(numeric)
    X_num_sparse = csr_matrix(X_num)

    # Combine
    hybrid = hstack([X_text, X_num_sparse], format="csr")
    hybrid = normalize(hybrid, norm="l2", axis=1)

    return hybrid, vectorizer, scaler


# ------------- PUBLIC: GET RECOMMENDATIONS -------------
def get_recommendations(
    data_path,
    food_type="Any",
    meal_types=None,
    price_min=0.0,
    price_max=1e9,
    dislike_words=None,
    top_k: int = TOP_K_DEFAULT,
) -> pd.DataFrame:
    """
    Main function used by Flask.

    Steps:
      1. Load + clean dataset
      2. Apply filters (type, meal types, price range, dislikes)
      3. Build hybrid ML matrix
      4. Compute centroid of filtered items
      5. Rank by cosine similarity to centroid
      6. Return top_k rows (as DataFrame)
    """
    if meal_types is None:
        meal_types = []
    if dislike_words is None:
        dislike_words = []

    # Load data
    df_all = load_dataset(data_path)

    # Apply filters
    filtered = _filter_dataset(
        df_all,
        food_type=food_type,
        meal_types=meal_types,
        price_min=price_min,
        price_max=price_max,
        dislike_words=dislike_words,
    )

    if filtered.empty:
        # Return empty DataFrame with same columns
        return filtered

    # Build hybrid matrix
    hybrid, _, _ = _build_hybrid_matrix(filtered)

    # Centroid of all filtered items
    centroid = hybrid.mean(axis=0)

    # Convert centroid to plain 2D numpy array
    if hasattr(centroid, "A"):           # numpy.matrix / sparse
        centroid_arr = np.asarray(centroid.A).reshape(1, -1)
    elif hasattr(centroid, "toarray"):   # sparse matrix
        centroid_arr = np.asarray(centroid.toarray()).reshape(1, -1)
    else:
        centroid_arr = np.asarray(centroid).reshape(1, -1)

    # Cosine similarity
    sims = cosine_similarity(hybrid, centroid_arr).ravel()  # length = n_filtered

    # Attach scores & sort
    filtered = filtered.reset_index(drop=True).copy()
    filtered["_sim_to_centroid"] = sims
    results = filtered.sort_values(
        by="_sim_to_centroid",
        ascending=False
    ).reset_index(drop=True)

    # Limit to top_k
    if top_k is None or top_k <= 0:
        top_k = TOP_K_DEFAULT
    results = results.head(top_k).reset_index(drop=True)

    return results

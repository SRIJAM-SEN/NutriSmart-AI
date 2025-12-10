from flask import Flask, render_template, request
from pathlib import Path
import os
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

from recommender_core import (
    load_dataset,
    get_recommendations,
    TOP_K_DEFAULT,
)

app = Flask(__name__)

# ----------------- CONFIG -----------------

# Base directory of this app.py file
BASE_DIR = Path(__file__).resolve().parent

# 🔹 Food recommendation CSV (now relative path for Render + local)
DATA_PATH_RECOMMENDER = BASE_DIR / "food_recommendation.csv"

# 🔹 7-days meal generator CSV (now relative path)
DATA_PATH_MEAL = BASE_DIR / "7_Days_Meal_Generator.csv"

# Load recommendation dataset once
df_all = load_dataset(DATA_PATH_RECOMMENDER)

# Choices for filters
type_values = sorted(df_all["type"].replace("", pd.NA).dropna().unique().tolist())
type_options = ["Any"] + type_values

meal_values = sorted(df_all["meal_type"].replace("", pd.NA).dropna().unique().tolist())

price_min_global = float(df_all["_price_f"].min()) if len(df_all) > 0 else 0.0
price_max_global = float(df_all["_price_f"].max()) if len(df_all) > 0 else price_min_global + 100.0


# ----------------- HOME PAGE -----------------

@app.route("/")
def home():
    return render_template("home.html")


# ----------------- FOOD RECOMMENDER PAGE -----------------

@app.route("/recommender", methods=["GET", "POST"])
def recommender():
    foods = []
    message = None

    selected_type = "Any"
    selected_meals = meal_values.copy()
    price_min = price_min_global
    price_max = price_max_global
    dislikes_text = ""
    extra_info = ""
    top_k = TOP_K_DEFAULT

    if request.method == "POST":
        selected_type = request.form.get("food_type") or "Any"
        selected_meals = request.form.getlist("meal_type") or meal_values

        try:
            price_min = float(request.form.get("price_min") or price_min_global)
        except ValueError:
            price_min = price_min_global

        try:
            price_max = float(request.form.get("price_max") or price_max_global)
        except ValueError:
            price_max = price_max_global

        dislikes_text = request.form.get("dislikes") or ""
        extra_info = request.form.get("extra_info") or ""
        try:
            top_k = int(request.form.get("top_k") or TOP_K_DEFAULT)
        except ValueError:
            top_k = TOP_K_DEFAULT

        dislike_words = [w.strip() for w in dislikes_text.split(",") if w.strip()]

        df_results = get_recommendations(
            data_path=DATA_PATH_RECOMMENDER,
            food_type=selected_type,
            meal_types=selected_meals,
            price_min=price_min,
            price_max=price_max,
            dislike_words=dislike_words,
            top_k=top_k,
        )

        if df_results.empty:
            message = "No items match your filters. Try relaxing filters or clearing dislikes."
        else:
            foods = []
            for _, row in df_results.iterrows():
                foods.append({
                    "Food_Name": row.get("Food_Name", row.get("food_name", "")),
                    "Food_Image_URL": row.get("Food_Image_URL", row.get("image_url", "")),
                    "Calories": row.get("Calories", row.get("_calories_f", "")),
                    "Protein": row.get("Protein", row.get("_protein_f", "")),
                    "Fat": row.get("Fat", row.get("_fat_f", "")),
                    "Carbs": row.get("Carbs", row.get("_carbs_f", "")),
                    "Price": row.get("Price", row.get("_price_f", "")),
                    "Cuisine": row.get("Cuisine", row.get("cuisine", "")),
                })

    return render_template(
        "recommender.html",
        foods=foods,
        message=message,
        type_options=type_options,
        meal_values=meal_values,
        selected_type=selected_type,
        selected_meals=selected_meals,
        price_min=price_min,
        price_max=price_max,
        price_min_global=price_min_global,
        price_max_global=price_max_global,
        dislikes_text=dislikes_text,
        extra_info=extra_info,
        top_k=top_k,
    )


# ----------------- 7 DAYS MEAL PLANNER LOGIC -----------------

PER_DAY_MEALS = ["Breakfast", "Lunch", "Snack", "Dinner"]
MEAL_SHARES = {"Breakfast": 0.25, "Lunch": 0.35, "Snack": 0.10, "Dinner": 0.30}


@lru_cache(maxsize=1)
def load_meal_data():
    if not DATA_PATH_MEAL.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH_MEAL}")
    df = pd.read_csv(DATA_PATH_MEAL)

    # cleanup BOM and whitespace
    df.columns = df.columns.str.strip().str.replace("﻿", "", regex=True)

    # numeric columns
    for col in ["Calories", "Protein", "Fat", "Carbs", "Price"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # text columns
    for col in ["Food_Name", "Type", "Meal_Type", "Cuisine"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    df["Ingredients"] = df["Food_Name"].fillna("").astype(str).str.lower()
    return df


def filter_foods(df, food_type, dislikes):
    if food_type:
        filtered = df[df["Type"].str.lower() == str(food_type).lower()].copy()
    else:
        filtered = df.copy()

    for item in dislikes:
        item = item.strip().lower()
        if item:
            mask = ~filtered["Food_Name"].str.contains(item, case=False, na=False) & \
                   ~filtered["Ingredients"].str.contains(item, case=False, na=False)
            filtered = filtered[mask]
    return filtered


def build_nutrition_matrix(df):
    features = df[["Calories", "Protein", "Fat", "Carbs"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return X, scaler


def pick_closest_for_meal(meal_options_df, scaler, target_calories):
    if meal_options_df.empty:
        return None
    feats = meal_options_df[["Calories", "Protein", "Fat", "Carbs"]].values
    Xc = scaler.transform(feats)

    calorie_dist = np.abs(meal_options_df["Calories"].values - target_calories)
    k = min(10, len(calorie_dist))
    shortlist_idx = np.argsort(calorie_dist)[:k]

    med = np.median(meal_options_df[["Protein", "Fat", "Carbs"]].values, axis=0)
    target_vec = np.array([target_calories, med[0], med[1], med[2]]).reshape(1, -1)
    Xt = scaler.transform(target_vec)

    dists = np.linalg.norm(Xc[shortlist_idx] - Xt, axis=1)
    pick_local = shortlist_idx[np.argmin(dists)]
    return meal_options_df.iloc[pick_local]


def random_pick(meal_options_df):
    if meal_options_df.empty:
        return None
    return meal_options_df.sample(1).iloc[0]


def generate_day_meal(food_df, max_calories, scaler, use_ml=True, tolerance=0.15):
    meals = {m: None for m in PER_DAY_MEALS}
    total_calories = 0.0

    for meal_type in PER_DAY_MEALS:
        opts = food_df[food_df["Meal_Type"].str.lower() == meal_type.lower()]
        if opts.empty:
            meals[meal_type] = None
            continue

        target = max_calories * MEAL_SHARES.get(meal_type, 0.25)
        if use_ml:
            pick = pick_closest_for_meal(opts, scaler, target) or random_pick(opts)
        else:
            pick = random_pick(opts)

        meals[meal_type] = pick
        if pick is not None:
            total_calories += float(pick["Calories"])

    lower = max_calories * (1 - tolerance)
    upper = max_calories * (1 + tolerance)
    attempts = 0
    while (total_calories < lower or total_calories > upper) and attempts < 80:
        candidates = [mt for mt in PER_DAY_MEALS if meals[mt] is not None]
        if not candidates:
            break
        mt = random.choice(candidates)
        opts = food_df[food_df["Meal_Type"].str.lower() == mt.lower()]
        if opts.empty:
            attempts += 1
            continue
        target = max_calories * MEAL_SHARES.get(mt, 0.25)
        if use_ml:
            candidate = pick_closest_for_meal(opts, scaler, target) or random_pick(opts)
        else:
            candidate = random_pick(opts)
        total_calories -= float(meals[mt]["Calories"])
        meals[mt] = candidate
        total_calories += float(candidate["Calories"])
        attempts += 1

    return meals, total_calories


def generate_meal_plan_week(food_df, max_calories, use_ml=True):
    X, scaler = build_nutrition_matrix(food_df)
    week = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day_name in days:
        meals, tot = generate_day_meal(food_df, max_calories, scaler, use_ml=use_ml)
        week.append({"Day": day_name, "Meals": meals, "Total_Calories": tot})
    return week


# ----------------- 7 DAYS MEAL PLANNER PAGE -----------------

@app.route("/meal-plan", methods=["GET", "POST"])
def meal_plan():
    df_meal = load_meal_data()

    food_types = sorted(df_meal["Type"].dropna().unique().tolist())
    if not food_types:
        food_types = [""]

    selected_type = ""
    max_calories = 2000
    dislikes_text = ""
    use_ml = True
    plan = []
    message = None

    if request.method == "POST":
        selected_type = request.form.get("food_type", "")
        max_calories = int(request.form.get("max_calories", 2000))
        dislikes_text = request.form.get("dislikes", "")
        use_ml = request.form.get("use_ml") == "on"

        dislikes_list = [d.strip() for d in dislikes_text.split(",") if d.strip()]
        filtered = filter_foods(df_meal, selected_type, dislikes_list)

        if filtered.empty:
            message = "No items left after applying filters/dislikes. Try changing Food Type or clearing dislikes."
        else:
            week_plan = generate_meal_plan_week(filtered, max_calories, use_ml=use_ml)
            plan = []
            for day in week_plan:
                meals_clean = {}
                for mt, food in day["Meals"].items():
                    if food is None:
                        meals_clean[mt] = None
                    else:
                        meals_clean[mt] = {
                            "Food_Name": food["Food_Name"],
                            "Food_Image_URL": food.get("Food_Image_URL", ""),
                            "Calories": float(food["Calories"]),
                            "Protein": float(food["Protein"]),
                            "Fat": float(food["Fat"]),
                            "Carbs": float(food["Carbs"]),
                            "Cuisine": food.get("Cuisine", ""),
                        }
                plan.append({
                    "Day": day["Day"],
                    "Meals": meals_clean,
                    "Total_Calories": day["Total_Calories"],
                })

    return render_template(
        "meal_plan.html",
        food_types=food_types,
        selected_type=selected_type,
        max_calories=max_calories,
        dislikes_text=dislikes_text,
        use_ml=use_ml,
        plan=plan,
        message=message,
    )


# ----------------- MAIN ENTRY (Render friendly) -----------------

if __name__ == "__main__":
    # For local dev, you can still run: python app.py
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

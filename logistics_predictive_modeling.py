"""
Logistics Analytics - Week 4
Predictive Modeling and Optimization in Logistics Systems
============================================================

Scenario (continued from Weeks 1-3): Regional E-Commerce Distribution Network

Prediction problem: Forecast shipment transit time (days) for a given
order, based on distance, weight, carrier, and warehouse, so that more
accurate delivery-date promises can be given to customers and routing/
staffing decisions can be made proactively.

This script:
  1. Simulates the same style of shipment dataset used in Week 3
  2. Prepares features (encoding, train/test split)
  3. Trains and compares two models: Linear Regression (baseline) and
     Random Forest Regressor (ensemble)
  4. Evaluates both with MAE, RMSE, and R-squared, using cross-validation
  5. Performs a small hyperparameter tuning pass on the Random Forest
  6. Extracts feature importance to guide optimization strategy
  7. Demonstrates a simple route-optimization heuristic (nearest-neighbor)
     informed by the predictive model's insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
OUT_DIR = "charts"

# ---------------------------------------------------------------------
# 1. PROBLEM DEFINITION & DATA SIMULATION
# ---------------------------------------------------------------------
def simulate_dataset(n=1500, seed=11):
    """
    Target variable : transit_days (continuous) - time from shipment
                       to delivery.
    Features        : distance_km, weight_kg, carrier, warehouse_id,
                       promised_days
    """
    rng = np.random.default_rng(seed)

    warehouses = rng.choice(["WH_A", "WH_B", "WH_C", "WH_D"], size=n,
                             p=[0.35, 0.25, 0.20, 0.20])
    carriers = rng.choice(["FedEx", "UPS", "DHL", "USPS"], size=n)

    warehouse_distance_bias = {"WH_A": 180, "WH_B": 260, "WH_C": 320, "WH_D": 220}
    distance_km = np.array([
        rng.normal(loc=warehouse_distance_bias[w], scale=60) for w in warehouses
    ]).clip(20, None).round(1)

    weight_kg = rng.gamma(shape=2.0, scale=5.0, size=n).round(2)

    carrier_speed_penalty = {"FedEx": 0.0, "UPS": 0.3, "DHL": 0.6, "USPS": 1.2}
    transit_days = (
        1 + distance_km / 150
        + np.array([carrier_speed_penalty[c] for c in carriers])
        + weight_kg * 0.01
        + rng.normal(0, 0.5, n)
    ).clip(1, None)

    promised_days = rng.integers(2, 7, size=n)

    df = pd.DataFrame({
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "carrier": carriers,
        "warehouse_id": warehouses,
        "promised_days": promised_days,
        "transit_days": transit_days.round(2),
    })
    return df


# ---------------------------------------------------------------------
# 2. MODEL SELECTION & IMPLEMENTATION
# ---------------------------------------------------------------------
def build_pipelines():
    """
    Builds two candidate pipelines:
      - Linear Regression: fast, interpretable baseline; assumes a
        roughly linear relationship between features and transit time.
      - Random Forest Regressor: ensemble of decision trees, chosen to
        capture potential non-linear interactions (e.g., carrier-specific
        distance effects) without heavy feature engineering.
    """
    categorical_features = ["carrier", "warehouse_id"]
    numeric_features = ["distance_km", "weight_kg", "promised_days"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
        remainder="passthrough",
    )

    lr_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ])

    return lr_pipeline, rf_pipeline, numeric_features, categorical_features


# ---------------------------------------------------------------------
# 3. EVALUATION & VALIDATION
# ---------------------------------------------------------------------
def evaluate_model(pipeline, X_test, y_test, name):
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} Test Performance ===")
    print(f"MAE:  {mae:.3f} days")
    print(f"RMSE: {rmse:.3f} days")
    print(f"R^2:  {r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}, preds


def cross_validate_model(pipeline, X, y, name, cv=5):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name} {cv}-fold CV MAE: {-scores.mean():.3f} (+/- {scores.std():.3f})")
    return -scores.mean()


def tune_random_forest(X_train, y_train, preprocessor):
    """Small grid search to tune key Random Forest hyperparameters."""
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3,
                         scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\nBest RF params: {grid.best_params_}")
    print(f"Best CV MAE: {-grid.best_score_:.3f}")
    return grid.best_estimator_


# ---------------------------------------------------------------------
# 4. FEATURE IMPORTANCE (for optimization insight)
# ---------------------------------------------------------------------
def plot_feature_importance(pipeline, numeric_features, categorical_features, X_train):
    model = pipeline.named_steps["model"]
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_names = cat_names + numeric_features

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(7, 5))
    plt.barh(imp_df["feature"], imp_df["importance"], color="#2E4057")
    plt.title("Random Forest Feature Importance - Transit Time Prediction")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_feature_importance.png", dpi=150)
    plt.close()
    return imp_df


def plot_predicted_vs_actual(y_test, preds, name, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.4, color="#4C7B8C")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1, label="Perfect Prediction")
    plt.title(f"Predicted vs. Actual Transit Time ({name})")
    plt.xlabel("Actual Transit Days")
    plt.ylabel("Predicted Transit Days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150)
    plt.close()


def plot_model_comparison(results_df):
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width / 2, results_df["MAE"], width, label="MAE", color="#2E4057")
    plt.bar(x + width / 2, results_df["RMSE"], width, label="RMSE", color="#4C7B8C")
    plt.xticks(x, results_df["model"])
    plt.ylabel("Error (days)")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_model_comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 5. OPTIMIZATION: ROUTE SEQUENCING HEURISTIC
# ---------------------------------------------------------------------
def nearest_neighbor_route(distance_matrix: np.ndarray, start: int = 0):
    """
    Simple nearest-neighbor heuristic for sequencing multiple deliveries
    from a single warehouse/depot, minimizing cumulative travel distance.
    Informed by the predictive model's finding that distance is the
    strongest driver of both cost and transit time.
    """
    n = distance_matrix.shape[0]
    visited = [start]
    unvisited = set(range(n)) - {start}
    total_distance = 0.0

    while unvisited:
        last = visited[-1]
        next_stop = min(unvisited, key=lambda j: distance_matrix[last, j])
        total_distance += distance_matrix[last, next_stop]
        visited.append(next_stop)
        unvisited.remove(next_stop)

    return visited, total_distance


def demo_route_optimization(seed=3):
    """Illustrative example: sequence 6 delivery stops from a depot."""
    rng = np.random.default_rng(seed)
    n_stops = 6
    coords = rng.uniform(0, 100, size=(n_stops, 2))
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    route, total_dist = nearest_neighbor_route(dist_matrix, start=0)
    print(f"\nOptimized route order (stop indices): {route}")
    print(f"Total route distance: {total_dist:.2f} units")
    plot_route(coords, route, total_dist)
    return coords, route, total_dist


def plot_route(coords, route, total_dist):
    plt.figure(figsize=(6, 6))
    ordered = coords[route]
    plt.plot(ordered[:, 0], ordered[:, 1], "o-", color="#2E4057", markersize=10)
    for i, (x, y) in enumerate(coords):
        label = "Depot" if i == route[0] else f"Stop {i}"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8))
    plt.title(f"Nearest-Neighbor Delivery Route (Total: {total_dist:.1f} units)")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_route_optimization.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = simulate_dataset()
    X = df.drop(columns=["transit_days"])
    y = df["transit_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_pipeline, rf_pipeline, numeric_features, categorical_features = build_pipelines()

    # --- Train baseline Linear Regression ---
    lr_pipeline.fit(X_train, y_train)
    lr_result, lr_preds = evaluate_model(lr_pipeline, X_test, y_test, "Linear Regression")
    cross_validate_model(lr_pipeline, X_train, y_train, "Linear Regression")

    # --- Train Random Forest ---
    rf_pipeline.fit(X_train, y_train)
    rf_result, rf_preds = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    cross_validate_model(rf_pipeline, X_train, y_train, "Random Forest")

    # --- Hyperparameter tuning ---
    preprocessor = rf_pipeline.named_steps["preprocess"]
    best_rf = tune_random_forest(X_train, y_train, preprocessor)
    best_result, best_preds = evaluate_model(best_rf, X_test, y_test, "Tuned Random Forest")

    # --- Visualizations ---
    imp_df = plot_feature_importance(best_rf, numeric_features, categorical_features, X_train)
    plot_predicted_vs_actual(y_test, lr_preds, "Linear Regression", "02_pred_vs_actual_lr.png")
    plot_predicted_vs_actual(y_test, best_preds, "Tuned Random Forest", "03_pred_vs_actual_rf.png")

    results_df = pd.DataFrame([lr_result, rf_result, best_result])
    plot_model_comparison(results_df)

    print("\n=== Feature Importance (Tuned Random Forest) ===")
    print(imp_df.sort_values("importance", ascending=False))

    # --- Optimization demo ---
    demo_route_optimization()

    print("\nAll charts saved to the 'charts/' directory.")

"""
Logistics Analytics Strategic Plan - Code Illustrations
Scenario: Regional E-Commerce Distribution Network
=========================================================

This script contains illustrative Python snippets (not production code)
showing the proposed end-to-end analytical approach described in the
strategic planning report:

    1. Data Collection & Ingestion
    2. Data Cleaning & Preparation
    3. Exploratory Data Analysis (EDA)
    4. KPI Computation
    5. Predictive Modeling (Demand Forecasting - Regression)
    6. Clustering (Warehouse / Delivery Zone Segmentation)
    7. Route Optimization (Heuristic / OR-based)

Each section is intentionally lightweight / pseudocode-like so that it
communicates approach and structure rather than a finished pipeline.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. DATA COLLECTION & INGESTION
# ---------------------------------------------------------------------
def load_datasets():
    """
    Load raw logistics data from multiple sources:
    - orders.csv        : order_id, sku, quantity, order_date, destination_zip
    - shipments.csv      : shipment_id, order_id, warehouse_id, carrier,
                            ship_date, delivery_date, cost
    - inventory.csv      : sku, warehouse_id, stock_level, reorder_point
    - warehouse_geo.csv  : warehouse_id, lat, lon, capacity
    """
    orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
    shipments = pd.read_csv(
        "data/shipments.csv", parse_dates=["ship_date", "delivery_date"]
    )
    inventory = pd.read_csv("data/inventory.csv")
    warehouses = pd.read_csv("data/warehouse_geo.csv")
    return orders, shipments, inventory, warehouses


# ---------------------------------------------------------------------
# 2. DATA CLEANING & PREPARATION
# ---------------------------------------------------------------------
def clean_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove duplicates, handle missing values,
    compute derived fields such as transit time."""
    df = shipments.drop_duplicates(subset="shipment_id")
    df = df.dropna(subset=["ship_date", "delivery_date"])

    # Remove impossible records (delivery before shipment)
    df = df[df["delivery_date"] >= df["ship_date"]]

    df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
    df["cost"] = df["cost"].fillna(df["cost"].median())
    return df


# ---------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------
def summarize_transit_performance(df: pd.DataFrame):
    """Return summary statistics used to understand delivery performance
    by carrier and warehouse."""
    summary = (
        df.groupby(["warehouse_id", "carrier"])["transit_days"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_transit_days"})
    )
    return summary


# ---------------------------------------------------------------------
# 4. KPI COMPUTATION
# ---------------------------------------------------------------------
def compute_kpis(orders, shipments, inventory):
    """
    Compute the three core KPIs identified in the strategic plan:
      1. On-Time Delivery Rate (OTD %)
      2. Average Order Fulfillment Cost
      3. Inventory Turnover Ratio
    """
    # --- On-Time Delivery Rate ---
    shipments["on_time"] = shipments["transit_days"] <= shipments["promised_days"]
    otd_rate = shipments["on_time"].mean() * 100

    # --- Average Order Fulfillment Cost ---
    avg_fulfillment_cost = shipments["cost"].mean()

    # --- Inventory Turnover Ratio (simplified) ---
    total_units_shipped = orders["quantity"].sum()
    avg_inventory = inventory["stock_level"].mean()
    inventory_turnover = total_units_shipped / avg_inventory

    return {
        "on_time_delivery_rate_pct": round(otd_rate, 2),
        "avg_fulfillment_cost": round(avg_fulfillment_cost, 2),
        "inventory_turnover_ratio": round(inventory_turnover, 2),
    }


# ---------------------------------------------------------------------
# 5. PREDICTIVE MODELING - DEMAND FORECASTING (Regression)
# ---------------------------------------------------------------------
def forecast_demand(order_history: pd.DataFrame):
    """
    Pseudocode-level illustration of a regression-based demand forecast
    per SKU using simple time-based and lag features.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # Feature engineering (illustrative)
    df = order_history.copy()
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["lag_7"] = df.groupby("sku")["quantity"].shift(7)
    df = df.dropna()

    features = ["day_of_week", "month", "lag_7"]
    X = df[features]
    y = df["quantity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print(f"Demand forecast MAE: {mae:.2f} units")
    return model


# ---------------------------------------------------------------------
# 6. CLUSTERING - DELIVERY ZONE / WAREHOUSE SEGMENTATION
# ---------------------------------------------------------------------
def cluster_delivery_zones(zone_features: pd.DataFrame, n_clusters: int = 4):
    """
    Group delivery zones by demand density, average distance to nearest
    warehouse, and average cost-to-serve, using K-Means. This supports
    warehouse placement and resource allocation decisions.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        zone_features[["demand_density", "avg_distance_km", "cost_to_serve"]]
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    zone_features["cluster"] = kmeans.fit_predict(scaled)
    return zone_features


# ---------------------------------------------------------------------
# 7. ROUTE OPTIMIZATION (Heuristic illustration)
# ---------------------------------------------------------------------
def nearest_neighbor_route(distance_matrix: np.ndarray, start: int = 0):
    """
    Simple nearest-neighbor heuristic for the Vehicle Routing / Traveling
    Salesman-style problem of sequencing deliveries from a depot.
    In production this would be replaced with an OR-Tools VRP solver
    that accounts for vehicle capacity and time windows.
    """
    n = distance_matrix.shape[0]
    visited = [start]
    unvisited = set(range(n)) - {start}

    while unvisited:
        last = visited[-1]
        next_stop = min(unvisited, key=lambda j: distance_matrix[last, j])
        visited.append(next_stop)
        unvisited.remove(next_stop)

    return visited


# ---------------------------------------------------------------------
# MAIN PIPELINE (illustrative orchestration)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    orders, shipments, inventory, warehouses = load_datasets()
    shipments = clean_shipments(shipments)

    transit_summary = summarize_transit_performance(shipments)
    kpis = compute_kpis(orders, shipments, inventory)

    print("=== KPI Summary ===")
    for k, v in kpis.items():
        print(f"{k}: {v}")

    # Forecasting and clustering would be called here with real data:
    # model = forecast_demand(orders)
    # zone_clusters = cluster_delivery_zones(zone_features)

"""
Logistics Analytics - Week 2
Data Collection, Cleaning, and Preprocessing Pipeline
======================================================

Scenario (continued from Week 1): Regional E-Commerce Distribution Network
Reference dataset type: Shipment / delivery records
(structurally similar to public datasets such as the Kaggle "Supply Chain
Shipment Pricing" dataset or DOT freight shipment data: order_id, sku,
warehouse_id, ship_date, delivery_date, promised_days, weight_kg,
distance_km, cost, carrier).

This script demonstrates, end-to-end, how raw shipment data is collected,
inspected, cleaned, and normalized before being handed off to the
analytical/modeling stage described in the Week 1 report.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. DATA COLLECTION (SIMULATED)
# ---------------------------------------------------------------------
def simulate_raw_shipment_data(n=1000, seed=42):
    """
    Simulates a raw shipment dataset with the kinds of quality issues
    typically found in real-world logistics data:
      - missing values
      - outliers (extreme costs / distances)
      - duplicate records
      - inconsistent categorical labels (e.g. 'fedex' vs 'FedEx')
    """
    rng = np.random.default_rng(seed)

    n_valid = int(n * 0.9)
    order_ids = np.arange(1, n + 1)
    warehouses = rng.choice(["WH_A", "WH_B", "WH_C", "WH_D"], size=n)
    carriers_clean = rng.choice(["FedEx", "UPS", "DHL", "USPS"], size=n)

    # introduce inconsistent capitalization for ~15% of records
    carriers = [
        c.lower() if rng.random() < 0.15 else c for c in carriers_clean
    ]

    distance_km = rng.normal(loc=250, scale=80, size=n).round(1)
    weight_kg = rng.normal(loc=12, scale=5, size=n).round(2)
    cost = (distance_km * 0.35 + weight_kg * 1.2 + rng.normal(0, 5, n)).round(2)

    ship_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 180, size=n), unit="D"
    )
    transit_days = rng.integers(1, 10, size=n)
    delivery_date = ship_date + pd.to_timedelta(transit_days, unit="D")
    promised_days = rng.integers(2, 7, size=n)

    df = pd.DataFrame({
        "order_id": order_ids,
        "warehouse_id": warehouses,
        "carrier": carriers,
        "ship_date": ship_date,
        "delivery_date": delivery_date,
        "promised_days": promised_days,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "cost": cost,
    })

    # --- Inject missing values (~5% of cost and weight_kg) ---
    for col in ["cost", "weight_kg"]:
        missing_idx = rng.choice(df.index, size=int(n * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan

    # --- Inject outliers (~1% extreme cost and distance values) ---
    outlier_idx = rng.choice(df.index, size=int(n * 0.01), replace=False)
    df.loc[outlier_idx, "cost"] = df.loc[outlier_idx, "cost"] * 20
    df.loc[outlier_idx, "distance_km"] = df.loc[outlier_idx, "distance_km"] * 10

    # --- Inject duplicate records (~2%) ---
    dup_rows = df.sample(int(n * 0.02), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ---------------------------------------------------------------------
# 2. DATA QUALITY ASSESSMENT
# ---------------------------------------------------------------------
def assess_data_quality(df: pd.DataFrame) -> dict:
    """Produce a quick data-quality report: missing values, duplicates,
    and basic descriptive statistics used to spot potential outliers."""
    report = {
        "n_rows": len(df),
        "n_duplicates": df.duplicated(subset="order_id").sum(),
        "missing_values": df.isna().sum().to_dict(),
        "cost_describe": df["cost"].describe().to_dict(),
        "distance_describe": df["distance_km"].describe().to_dict(),
    }
    return report


# ---------------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------------
def clean_shipment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a sequence of cleaning steps:
      1. Remove exact/order_id duplicates
      2. Standardize categorical text (carrier names)
      3. Handle missing values via median imputation
      4. Detect and treat outliers using the IQR method
      5. Validate logical consistency (delivery_date >= ship_date)
    """
    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates(subset="order_id", keep="first")

    # 2. Standardize categorical labels
    df["carrier"] = df["carrier"].str.strip().str.title()

    # 3. Handle missing values (median imputation for numeric columns)
    for col in ["cost", "weight_kg"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 4. Outlier detection & treatment using the IQR method
    for col in ["cost", "distance_km"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # Cap outliers instead of dropping rows, to preserve sample size
        df[col] = df[col].clip(lower=lower, upper=upper)

    # 5. Logical consistency check
    df = df[df["delivery_date"] >= df["ship_date"]]

    # Derived field used downstream
    df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# 4. NORMALIZATION / SCALING
# ---------------------------------------------------------------------
def normalize_features(df: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Applies Min-Max normalization to numeric columns so that features
    with different scales (e.g., cost in dollars vs. distance in km)
    contribute proportionately to downstream models such as clustering
    or regression.
    """
    df = df.copy()
    for col in columns:
        min_val, max_val = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
    return df


def standardize_features(df: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Alternative to Min-Max scaling: Z-score standardization
    (mean = 0, std = 1). Preferred when the downstream model assumes
    normally distributed inputs (e.g., linear regression coefficients).
    """
    df = df.copy()
    for col in columns:
        mean_val, std_val = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mean_val) / std_val
    return df


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = simulate_raw_shipment_data()
    print("=== Raw Data Quality Report ===")
    quality_report = assess_data_quality(raw_df)
    print(f"Rows: {quality_report['n_rows']}")
    print(f"Duplicate order_ids: {quality_report['n_duplicates']}")
    print(f"Missing values: {quality_report['missing_values']}")

    cleaned_df = clean_shipment_data(raw_df)
    cleaned_df = normalize_features(cleaned_df, ["cost", "distance_km", "weight_kg"])
    cleaned_df = standardize_features(cleaned_df, ["cost", "distance_km"])

    print("\n=== Cleaned Data Sample ===")
    print(cleaned_df.head())

    print(f"\nRows before cleaning: {len(raw_df)}")
    print(f"Rows after cleaning:  {len(cleaned_df)}")

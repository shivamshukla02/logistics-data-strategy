"""
Logistics Analytics - Week 3
Advanced Data Analysis and Visualization
==========================================

Scenario (continued from Weeks 1-2): Regional E-Commerce Distribution Network

This script simulates a cleaned logistics dataset (as would result from the
Week 2 preprocessing pipeline) and performs exploratory data analysis (EDA)
and visualization to surface insights about delivery performance, cost
drivers, and operational bottlenecks.

Outputs:
  - Prints EDA summary statistics and correlation matrix to console
  - Saves 6 chart images (PNG) to the charts/ directory for inclusion
    in the Week 3 report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT_DIR = "charts"

# ---------------------------------------------------------------------
# 1. DATA SIMULATION (represents the cleaned Week 2 output)
# ---------------------------------------------------------------------
def simulate_clean_dataset(n=1200, seed=7):
    rng = np.random.default_rng(seed)

    warehouses = rng.choice(["WH_A", "WH_B", "WH_C", "WH_D"], size=n,
                             p=[0.35, 0.25, 0.20, 0.20])
    carriers = rng.choice(["FedEx", "UPS", "DHL", "USPS"], size=n)

    # Base distance varies by warehouse (some warehouses serve farther zones)
    warehouse_distance_bias = {"WH_A": 180, "WH_B": 260, "WH_C": 320, "WH_D": 220}
    distance_km = np.array([
        rng.normal(loc=warehouse_distance_bias[w], scale=60) for w in warehouses
    ]).clip(20, None).round(1)

    weight_kg = rng.gamma(shape=2.0, scale=5.0, size=n).round(2)

    # Carrier cost efficiency differs slightly
    carrier_rate = {"FedEx": 0.42, "UPS": 0.39, "DHL": 0.47, "USPS": 0.33}
    base_cost = np.array([carrier_rate[c] for c in carriers]) * distance_km
    cost = (base_cost + weight_kg * 1.1 + rng.normal(0, 4, n)).clip(5, None).round(2)

    # Transit/delivery time correlates with distance, plus carrier variability
    carrier_speed_penalty = {"FedEx": 0.0, "UPS": 0.3, "DHL": 0.6, "USPS": 1.2}
    transit_days = (
        1 + distance_km / 150
        + np.array([carrier_speed_penalty[c] for c in carriers])
        + rng.normal(0, 0.6, n)
    ).clip(1, None).round().astype(int)

    promised_days = rng.integers(2, 7, size=n)
    on_time = transit_days <= promised_days

    order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 270, size=n), unit="D"
    )

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "warehouse_id": warehouses,
        "carrier": carriers,
        "order_date": order_date,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "cost": cost,
        "transit_days": transit_days,
        "promised_days": promised_days,
        "on_time": on_time,
    })
    return df


# ---------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("=== Central Tendency & Spread ===")
    print(df[["distance_km", "weight_kg", "cost", "transit_days"]].describe().round(2))

    print("\n=== On-Time Delivery Rate by Carrier ===")
    print((df.groupby("carrier")["on_time"].mean() * 100).round(1))

    print("\n=== Average Cost by Warehouse ===")
    print(df.groupby("warehouse_id")["cost"].mean().round(2))

    print("\n=== Correlation Matrix ===")
    corr = df[["distance_km", "weight_kg", "cost", "transit_days"]].corr()
    print(corr.round(2))
    return corr


# ---------------------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------------------
def plot_transit_distribution(df):
    """Histogram: shows the shape/spread of delivery (transit) times."""
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df["transit_days"], discrete=True, kde=False, color="#2E4057")
    plt.title("Distribution of Shipment Transit Times")
    plt.xlabel("Transit Days")
    plt.ylabel("Number of Shipments")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_transit_time_distribution.png", dpi=150)
    plt.close()


def plot_cost_by_carrier(df):
    """Boxplot: compares cost distribution and variability across carriers."""
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="carrier", y="cost", hue="carrier",
                palette="Set2", legend=False)
    plt.title("Shipping Cost Distribution by Carrier")
    plt.xlabel("Carrier")
    plt.ylabel("Cost (USD)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_cost_by_carrier_boxplot.png", dpi=150)
    plt.close()


def plot_distance_vs_cost(df):
    """Scatter plot: examines the relationship between distance and cost."""
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="distance_km", y="cost", hue="carrier",
                     alpha=0.6, palette="Set2")
    plt.title("Shipping Distance vs. Cost")
    plt.xlabel("Distance (km)")
    plt.ylabel("Cost (USD)")
    plt.legend(title="Carrier", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_distance_vs_cost_scatter.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    """Heatmap: visualizes correlation strength among numeric variables."""
    corr = df[["distance_km", "weight_kg", "cost", "transit_days"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix: Key Logistics Variables")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_monthly_volume(df):
    """Line chart: tracks shipment volume trend over time."""
    monthly = df.set_index("order_date").resample("ME").size()
    plt.figure(figsize=(8, 4.5))
    plt.plot(monthly.index, monthly.values, marker="o", color="#2E4057")
    plt.title("Monthly Shipment Volume Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Shipments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_monthly_volume_trend.png", dpi=150)
    plt.close()


def plot_otd_by_warehouse(df):
    """Bar chart: compares on-time delivery rate across warehouses."""
    otd = (df.groupby("warehouse_id")["on_time"].mean() * 100).round(1)
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(otd.index, otd.values, color="#4C7B8C")
    plt.axhline(95, color="red", linestyle="--", linewidth=1, label="Target (95%)")
    plt.title("On-Time Delivery Rate by Warehouse")
    plt.xlabel("Warehouse")
    plt.ylabel("On-Time Delivery Rate (%)")
    plt.ylim(0, 100)
    plt.legend()
    for bar, val in zip(bars, otd.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_otd_by_warehouse.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = simulate_clean_dataset()
    run_eda(df)

    plot_transit_distribution(df)
    plot_cost_by_carrier(df)
    plot_distance_vs_cost(df)
    plot_correlation_heatmap(df)
    plot_monthly_volume(df)
    plot_otd_by_warehouse(df)

    print("\nAll charts saved to the 'charts/' directory.")

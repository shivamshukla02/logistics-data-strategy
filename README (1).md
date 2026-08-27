# Logistics Analytics Project – Weeks 1–4

## Overview
This repository documents a four-week logistics data analytics project, progressing from strategic planning through data preprocessing, exploratory analysis and visualization, and finally predictive modeling and optimization. Each week builds on the same scenario:

**Scenario:** A regional e-commerce company operating a multi-warehouse (WH_A–WH_D) distribution network, served by four carriers (FedEx, UPS, DHL, USPS), facing rising delivery costs and inconsistent on-time performance.

## Repository Structure
```
├── week1/
│   ├── Logistics_Strategic_Planning_Report.docx
│   └── logistics_analysis_pipeline.py
├── week2/
│   ├── Logistics_Data_Preprocessing_Report.docx
│   └── logistics_data_preprocessing.py
├── week3/
│   ├── Logistics_Analysis_Visualization_Report.docx
│   ├── logistics_eda_visualization.py
│   └── charts/
│       ├── 01_transit_time_distribution.png
│       ├── 02_cost_by_carrier_boxplot.png
│       ├── 03_distance_vs_cost_scatter.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_monthly_volume_trend.png
│       └── 06_otd_by_warehouse.png
├── week4/
│   ├── Logistics_Predictive_Modeling_Report.docx
│   ├── logistics_predictive_modeling.py
│   └── charts/
│       ├── 01_feature_importance.png
│       ├── 02_pred_vs_actual_lr.png
│       ├── 03_pred_vs_actual_rf.png
│       ├── 04_model_comparison.png
│       └── 05_route_optimization.png
└── README.md
```

## Week 1 – Strategic Planning and Data Exploration
Defines the logistics scenario and three core KPIs: **On-Time Delivery Rate** (target ≥95%), **Average Order Fulfillment Cost** (target: reduce 10%), and **Inventory Turnover Ratio** (target: increase 15%). Outlines an 8-phase analytical roadmap and illustrates the approach with Python code covering KPI computation, demand forecasting, and delivery-zone clustering.

## Week 2 – Data Collection, Cleaning, and Preprocessing
Simulates a raw shipment dataset with realistic quality issues — missing values, outliers, duplicate records, and inconsistent carrier labels — and builds a cleaning pipeline using median imputation, IQR-based outlier capping, deduplication, text standardization, and logical validation. Applies Min-Max normalization and Z-score standardization to prepare features for modeling.

## Week 3 – Advanced Data Analysis and Visualization
Performs exploratory data analysis on a cleaned 1,200-record shipment dataset, computing summary statistics and a correlation matrix (distance–cost correlation of 0.91). Produces six visualizations — histogram, boxplot, scatter plot, heatmap, trend line, and benchmark bar chart — each justified by its analytical purpose, and derives insights on carrier performance gaps and warehouse bottlenecks (notably WH_C).

## Week 4 – Predictive Modeling and Optimization
Builds and compares two predictive models (Linear Regression and Random Forest) to forecast shipment transit time, evaluated with MAE, RMSE, R², 5-fold cross-validation, and hyperparameter tuning via grid search. Extracts feature importance to confirm distance as the dominant driver of transit time, and demonstrates a nearest-neighbor route-optimization heuristic. Concludes with model-informed optimization recommendations: carrier reallocation, dynamic delivery-time promises, warehouse-to-zone reassignment, and route batching.

## Key Findings Across the Project
- **Distance** is the single strongest driver of both shipping cost (r = 0.91) and transit time.
- **Carrier performance varies significantly**: FedEx achieves the highest on-time delivery rate (89.4%), while USPS lags (67.2%) despite lower cost.
- **Warehouse WH_C** shows the highest average cost and lowest on-time delivery rate, making it the top candidate for network reconfiguration.
- A simple, interpretable **Linear Regression model** matched or outperformed a more complex Random Forest for transit-time prediction — a reminder that model complexity should be justified by measurable performance gains.

## Tools & Techniques
- **Python**: pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Techniques**: data cleaning & normalization, exploratory data analysis, regression & ensemble modeling, cross-validation & hyperparameter tuning, K-Means clustering, route optimization heuristics

## Status
Project complete through Week 4. Together, these reports form an end-to-end logistics analytics workflow — from strategic framing to a deployable predictive and optimization toolkit.

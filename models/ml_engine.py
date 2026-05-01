ML Models for Crime Prediction & Pattern Analysis
 
Models:
  1. RandomForest — Crime type classification
  2. KMeans — Hotspot clustering
  3. IsolationForest — Anomaly (crime spike) detection
  4. LinearRegression — Temporal trend forecasting
  5. GradientBoosting — Severity prediction
"""
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
 
 
@st.cache_resource(show_spinner=False)
def train_hotspot_clustering(df: pd.DataFrame, n_clusters: int = 8):
    """KMeans clustering to identify crime hotspots."""
    coords = df[["latitude", "longitude"]].dropna().values
    if len(coords) < n_clusters:
        return None, None, None
 
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_scaled)
 
    try:
        score = silhouette_score(coords_scaled, labels, sample_size=min(5000, len(coords_scaled)))
    except Exception:
        score = 0.0
 
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
 
    cluster_stats = []
    df_copy = df.copy()
    df_copy["cluster"] = labels
    for i in range(n_clusters):
        cluster_df = df_copy[df_copy["cluster"] == i]
        cluster_stats.append({
            "cluster_id": i,
            "center_lat": centers_original[i][0],
            "center_lon": centers_original[i][1],
            "count": len(cluster_df),
            "avg_severity": cluster_df["severity"].mean(),
            "top_crime": cluster_df["crime_type"].mode()[0] if len(cluster_df) > 0 else "Unknown",
            "risk_score": (len(cluster_df) / len(df_copy)) * cluster_df["severity"].mean() * 10,
        })
 
    return kmeans, pd.DataFrame(cluster_stats), score
 
 
@st.cache_resource(show_spinner=False)
def train_crime_classifier(df: pd.DataFrame):
    """RandomForest to classify crime type from spatiotemporal features."""
    df = df.copy()
    df = df.dropna(subset=["latitude", "longitude", "hour", "crime_type"])
 
    top_types = df["crime_type"].value_counts().head(10).index
    df = df[df["crime_type"].isin(top_types)]
 
    if len(df) < 100:
        return None, None, None, None
 
    le = LabelEncoder()
    df["crime_label"] = le.fit_transform(df["crime_type"])
 
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["day_num"] = df["day_of_week"].map(day_map).fillna(0).astype(int)
 
    features = ["latitude", "longitude", "hour", "day_num", "severity"]
    X = df[features].values
    y = df["crime_label"].values
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
 
    accuracy = clf.score(X_test, y_test)
    importance = dict(zip(features, clf.feature_importances_))
 
    return clf, le, accuracy, importance
 
 
@st.cache_resource(show_spinner=False)
def train_anomaly_detector(df: pd.DataFrame):
    """IsolationForest to detect unusual crime spikes."""
    daily = df.groupby("date_only").size().reset_index(name="count")
    daily["date_only"] = pd.to_datetime(daily["date_only"])
    daily = daily.sort_values("date_only")
 
    if len(daily) < 10:
        return None, daily
 
    daily["day_of_week"] = daily["date_only"].dt.dayofweek
    daily["rolling_mean"] = daily["count"].rolling(7, min_periods=1).mean()
    daily["rolling_std"] = daily["count"].rolling(7, min_periods=1).std().fillna(1)
 
    X = daily[["count", "day_of_week", "rolling_mean"]].values
    iso = IsolationForest(contamination=0.1, random_state=42)
    daily["anomaly"] = iso.fit_predict(X)
    daily["anomaly_score"] = iso.score_samples(X)
    daily["is_anomaly"] = daily["anomaly"] == -1
 
    return iso, daily
 
 
@st.cache_data(show_spinner=False)
def forecast_crime_trend(df: pd.DataFrame, forecast_days: int = 14):
    """Linear regression + rolling average for temporal forecasting."""
    daily = df.groupby("date_only").size().reset_index(name="count")
    daily["date_only"] = pd.to_datetime(daily["date_only"])
    daily = daily.sort_values("date_only")
 
    if len(daily) < 7:
        return pd.DataFrame()
 
    daily["day_num"] = (daily["date_only"] - daily["date_only"].min()).dt.days
    daily["rolling7"] = daily["count"].rolling(7, min_periods=1).mean()
 
    X = daily[["day_num"]].values
    y = daily["count"].values
 
    reg = LinearRegression()
    reg.fit(X, y)
 
    last_day = daily["day_num"].max()
    last_date = daily["date_only"].max()
 
    future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
    future_preds = reg.predict(future_days)
 
    # Add realistic noise
    noise = np.random.normal(0, daily["count"].std() * 0.15, size=forecast_days)
    future_preds = np.maximum(0, future_preds + noise)
 
    forecast_df = pd.DataFrame({
        "date_only": future_dates,
        "count": future_preds,
        "type": "forecast",
    })
 
    daily["type"] = "actual"
    combined = pd.concat([daily[["date_only", "count", "type"]], forecast_df], ignore_index=True)
    return combined, reg.coef_[0], reg.intercept_
 
 
@st.cache_resource(show_spinner=False)
def train_severity_predictor(df: pd.DataFrame):
    """GradientBoosting to predict crime severity from features."""
    df = df.copy().dropna(subset=["latitude", "longitude", "hour", "severity"])
 
    if len(df) < 100:
        return None, None
 
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["day_num"] = df["day_of_week"].map(day_map).fillna(0).astype(int)
 
    features = ["latitude", "longitude", "hour", "day_num"]
    X = df[features].values
    y = df["severity"].values
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
 
    score = gb.score(X_test, y_test)
    return gb, score
 
 
def predict_risk_grid(severity_model, lat_range: tuple, lon_range: tuple, resolution: int = 25):
    """Generate a risk grid for heatmap overlay."""
    lats = np.linspace(lat_range[0], lat_range[1], resolution)
    lons = np.linspace(lon_range[0], lon_range[1], resolution)
 
    grid_points = []
    for hour in [0, 6, 12, 18]:  # sample hours
        for lat in lats:
            for lon in lons:
                grid_points.append([lat, lon, hour, 4])  # day_num=4 (Friday)
 
    X_grid = np.array(grid_points)
    preds = severity_model.predict(X_grid)
 
    results = []
    for i, (lat, lon, hour, _) in enumerate(grid_points):
        results.append({"latitude": lat, "longitude": lon, "hour": int(hour), "risk": preds[i]})
 
    grid_df = pd.DataFrame(results)
    grid_df = grid_df.groupby(["latitude", "longitude"])["risk"].mean().reset_index()
    return grid_df

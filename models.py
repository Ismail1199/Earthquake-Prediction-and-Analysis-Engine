# models.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import pairwise_distances_argmin_min  # <-- NEW IMPORT FOR FIX

# ===================== LOAD DATA =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "earthquake.csv")
df = pd.read_csv(CSV_PATH)
df["time"] = pd.to_datetime(df["time"], errors="coerce")

num_cols = ["latitude","longitude","depth","mag","nst","gap","dmin","rms","horizontalError","depthError","magError","magNst"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[(df["mag"] > 0) & (df["depth"] >= 0)]
df.fillna(df.median(numeric_only=True), inplace=True)
df = df.sort_values("time")

# ===================== FEATURE ENGINEERING =====================
df["lat_bin"] = df["latitude"].round(1)
df["lon_bin"] = df["longitude"].round(1)

rows = []
for (_, _), g in df.groupby(["lat_bin", "lon_bin"]):
    g = g.sort_values("time")
    for i in range(len(g)):
        t = g.iloc[i]["time"]
        past = g[g["time"] >= t - pd.Timedelta(days=7)]
        future = g[(g["time"] > t) & (g["time"] <= t + pd.Timedelta(days=7))]
        if len(past) < 3:
            continue
        rows.append({
            "eq_count_7d": len(past),
            "max_mag_7d": past["mag"].max(),
            "avg_depth_7d": past["depth"].mean(),
            "future_strong": int((future["mag"] >= 5).any()),
            "future_avg_mag": future["mag"].mean()
        })

data = pd.DataFrame(rows).dropna()

# ===================== MODEL 1 — CLASSIFICATION =====================
X_cls = data[["eq_count_7d", "max_mag_7d", "avg_depth_7d"]]
y_cls = data["future_strong"]
clf = LogisticRegression(max_iter=1000)
clf.fit(X_cls, y_cls)
cls_acc = accuracy_score(y_cls, clf.predict(X_cls))

# ===================== MODEL 2 — REGRESSION =====================
X_reg = data[["eq_count_7d", "max_mag_7d", "avg_depth_7d"]]
y_reg = data["future_avg_mag"]
reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
reg.fit(X_reg, y_reg)
reg_pred = reg.predict(X_reg)
reg_r2 = r2_score(y_reg, reg_pred)
reg_mae = mean_absolute_error(y_reg, reg_pred)
reg_mse = mean_squared_error(y_reg, reg_pred)

# ===================== MODEL 3 — CLUSTERING =====================
coords = df[["latitude", "longitude"]]
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
dbscan = DBSCAN(eps=0.35, min_samples=40)
df["cluster"] = dbscan.fit_predict(coords_scaled)
cluster_count = len(set(df["cluster"])) - (1 if -1 in df["cluster"] else 0)

# ===================== PREDICTION FUNCTIONS =====================
def predict_strong(eq_count, max_mag, days_gap):
    """
    Model 1: Probability of ≥5.0 quake in next 7 days
    Inputs: eq_count, max_mag (last 7 days), days_gap
    """
    gap_feature = max(0, days_gap - 3)  # risk increases after 3 quiet days
    prob = clf.predict_proba([[eq_count, max_mag, gap_feature]])[0][1]
    return float(prob)

def predict_future_mag(eq_count, avg_mag, avg_depth):
    """
    Model 2: Predicted average magnitude in next 7 days
    Inputs: eq_count, avg_mag (last 7 days), avg_depth
    """
    pred = reg.predict([[eq_count, avg_mag, avg_depth]])[0]
    return float(pred)

def get_cluster(lat, lon):
    """
    Fixed version: Assign new point to nearest historical earthquake's cluster
    Returns cluster ID or -1 if nearest point is an outlier
    """
    point = scaler.transform([[lat, lon]])
    closest_idx, _ = pairwise_distances_argmin_min(point, coords_scaled)
    closest_cluster = int(df["cluster"].iloc[closest_idx[0]])
    return closest_cluster if closest_cluster != -1 else -1

# ===================== EDA & METRICS (when run directly) =====================
if __name__ == "__main__":
    print("\nMODEL METRICS\n")
    print("Model 1 – Classification (Strong Quake ≥5.0 in next 7 days)")
    print(f"Accuracy: {cls_acc:.3f}\n")

    print("Model 2 – Regression (Future Average Magnitude)")
    print(f"R²: {reg_r2:.3f}")
    print(f"MAE: {reg_mae:.3f}")
    print(f"MSE: {reg_mse:.3f}\n")

    print("Model 3 – Clustering")
    print(f"Clusters Found: {cluster_count}\n")

    # -------- VISUALS (4) --------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["mag"], bins=30, kde=True, color="#2563eb")
    plt.title("Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.show()

    plt.figure(figsize=(8, 5))
    df.resample("D", on="time").size().plot(color="#2563eb")
    plt.title("Earthquakes per Day")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="depth", y="mag", data=df, alpha=0.5, color="#2563eb")
    plt.title("Depth vs Magnitude")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="longitude",
        y="latitude",
        hue="cluster",
        data=df,
        palette="tab10",
        legend=False,
        alpha=0.7,
        size=5
    )
    plt.title("Seismic Clusters (DBSCAN)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    print("EDA visualizations displayed.")
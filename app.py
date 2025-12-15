from flask import Flask, render_template, request, jsonify
import models
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Folder to save generated plots
PLOTS_DIR = os.path.join(app.static_folder, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Generate and save the 4 plots once when the app starts
def generate_plots():
    plot_paths = {
        "mag_dist": os.path.join(PLOTS_DIR, "magnitude_distribution.png"),
        "daily_eq": os.path.join(PLOTS_DIR, "earthquakes_per_day.png"),
        "depth_mag": os.path.join(PLOTS_DIR, "depth_vs_magnitude.png"),
        "clusters": os.path.join(PLOTS_DIR, "seismic_clusters.png")
    }

    # Only generate if not already present
    if all(os.path.exists(path) for path in plot_paths.values()):
        return plot_paths

    plt.style.use("default")  # Clean, professional look

    # 1. Magnitude Distribution
    plt.figure(figsize=(10, 6))
    models.sns.histplot(models.df["mag"], bins=30, kde=True, color="#2563eb")
    plt.title("Magnitude Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_paths["mag_dist"])
    plt.close()

    # 2. Earthquakes per Day
    plt.figure(figsize=(10, 6))
    models.df.resample("D", on="time").size().plot(color="#2563eb", linewidth=2)
    plt.title("Earthquakes per Day", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Earthquakes")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(plot_paths["daily_eq"])
    plt.close()

    # 3. Depth vs Magnitude
    plt.figure(figsize=(10, 6))
    models.sns.scatterplot(x="depth", y="mag", data=models.df, alpha=0.6, color="#2563eb")
    plt.title("Depth vs Magnitude", fontsize=16, fontweight="bold")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(plot_paths["depth_mag"])
    plt.close()

    # 4. Seismic Clusters
    plt.figure(figsize=(12, 8))
    models.sns.scatterplot(
        x="longitude", y="latitude",
        hue="cluster", data=models.df,
        palette="tab10", legend=False,
        alpha=0.7, s=30
    )
    plt.title("Global Seismic Clusters (DBSCAN)", fontsize=16, fontweight="bold")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(plot_paths["clusters"])
    plt.close()

    print("All insight plots generated and saved!")
    return plot_paths


# Generate plots on startup
PLOT_PATHS = generate_plots()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/insights")
def insights():
    # No need to pass plot paths anymore — images are linked directly via url_for()
    return render_template("insights.html")


# --- Prediction API Endpoints (with error handling) ---
@app.route("/predict/strong", methods=["POST"])
def predict_strong():
    try:
        data = request.get_json()
        prob = models.predict_strong(
            float(data["eq_count"]),
            float(data["max_mag"]),
            float(data["days_gap"])
        )
        level = "high" if prob >= 0.6 else "medium" if prob >= 0.3 else "low"
        return jsonify({"probability": round(prob, 3), "level": level})
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input"}), 400


@app.route("/predict/magnitude", methods=["POST"])
def predict_magnitude():
    try:
        data = request.get_json()
        pred = models.predict_future_mag(
            float(data["eq_count"]),
            float(data["avg_mag"]),
            float(data["avg_depth"])
        )
        return jsonify({"prediction": round(pred, 2)})
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input"}), 400


@app.route("/predict/zone", methods=["POST"])
def predict_zone():
    try:
        data = request.get_json()
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "Coordinates out of range"}), 400
        zone = models.get_cluster(lat, lon)
        status = "Active Seismic Cluster" if zone != -1 else "Low Activity / Outlier Zone"
        return jsonify({"zone": zone, "status": status})
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400


if __name__ == "__main__":
    app.run(debug=True)
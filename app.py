# ============================================================
# app.py — House Price Prediction: Flask Web Application
# ============================================================
# This file creates a web server. When users fill the form
# on the website and click "Predict", this code:
#   1. Receives the form data
#   2. Preprocesses it (same way we did during training)
#   3. Feeds it into our saved ML model
#   4. Returns the predicted price
# ============================================================

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

# Create the Flask app
app = Flask(__name__)

# ----------------------------------------------------------
# Load the trained model (we saved it in train_model.py)
# ----------------------------------------------------------
MODEL_PATH   = "model/house_price_model.pkl"
FEATURES_PATH = "model/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found! Please run: python train_model.py"
    )

model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
print(f"✓ Model loaded. Features: {feature_names}")


# ----------------------------------------------------------
# Route 1: Home Page — serves the HTML form
# ----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------------------------------
# Route 2: Prediction API — receives form data, returns price
# ----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Read form inputs ──────────────────────────────
        data = request.get_json()

        area       = float(data["area"])
        bedrooms   = int(data["bedrooms"])
        bathrooms  = int(data["bathrooms"])
        stories    = int(data["stories"])
        parking    = int(data["parking"])

        # yes/no fields → 1/0
        mainroad       = 1 if data["mainroad"]       == "yes" else 0
        guestroom      = 1 if data["guestroom"]      == "yes" else 0
        basement       = 1 if data["basement"]       == "yes" else 0
        hotwaterheating= 1 if data["hotwaterheating"]== "yes" else 0
        airconditioning= 1 if data["airconditioning"]== "yes" else 0
        prefarea       = 1 if data["prefarea"]       == "yes" else 0

        # furnishingstatus → 2/1/0
        furnish_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
        furnishingstatus = furnish_map.get(data["furnishingstatus"], 1)

        # ── Feature Engineering (MUST match train_model.py) ──
        area_per_bedroom = area / (bedrooms + 1)
        total_rooms      = bedrooms + bathrooms
        luxury_score     = (airconditioning + hotwaterheating
                            + prefarea + furnishingstatus)
        area_bath        = area * bathrooms
        stories_bath     = stories * bathrooms

        # ── Build the feature vector in the SAME ORDER as training ──
        feature_vector = [
            area, bedrooms, bathrooms, stories,
            mainroad, guestroom, basement,
            hotwaterheating, airconditioning, parking,
            prefarea, furnishingstatus,
            area_per_bedroom, total_rooms, luxury_score,
            area_bath, stories_bath
        ]

        # ── Make Prediction ──────────────────────────────
        input_array    = np.array([feature_vector])
        predicted_price = model.predict(input_array)[0]

        # Clamp to a realistic range (prices shouldn't be negative)
        predicted_price = max(500_000, predicted_price)

        return jsonify({
            "success": True,
            "predicted_price": round(predicted_price),
            "formatted_price": f"₹{predicted_price:,.0f}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ----------------------------------------------------------
# Entry point — start the server
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 45)
    print("  🏠 House Price Predictor — Web App")
    print("=" * 45)
    print("  Open your browser and go to:")
    print("  👉  http://127.0.0.1:5000")
    print("=" * 45 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

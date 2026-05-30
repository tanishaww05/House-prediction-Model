# ============================================================
# train_model.py — House Price Prediction: Model Training
# ============================================================
# Run this script ONCE to train the model and save it to disk.
# After running, app.py uses the saved model to make predictions.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("=" * 55)
print("   HOUSE PRICE PREDICTION — MODEL TRAINING")
print("=" * 55)

# ----------------------------------------------------------
# STEP 1: Load the Dataset
# ----------------------------------------------------------
print("\n[1/5] Loading dataset...")
df = pd.read_csv("Housing.csv")
print(f"      ✓ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# ----------------------------------------------------------
# STEP 2: Data Preprocessing (Encoding)
# ----------------------------------------------------------
# Machine learning models only understand numbers.
# Columns like "yes/no" and "furnished" need to be turned
# into numbers. This process is called ENCODING.
print("\n[2/5] Preprocessing data...")

# Binary columns: yes → 1, no → 0
binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Ordinal encoding: furnished=2, semi-furnished=1, unfurnished=0
df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})
print("      ✓ Encoded categorical columns to numbers")

# ----------------------------------------------------------
# STEP 3: Feature Engineering
# ----------------------------------------------------------
# Feature engineering = creating NEW useful columns from
# existing ones. This helps the model learn better patterns.
print("\n[3/5] Engineering features...")

df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
df['total_rooms']      = df['bedrooms'] + df['bathrooms']
df['luxury_score']     = (df['airconditioning'] + df['hotwaterheating']
                          + df['prefarea'] + df['furnishingstatus'])
df['area_bath']        = df['area'] * df['bathrooms']
df['stories_bath']     = df['stories'] * df['bathrooms']

print(f"      ✓ Created 5 new features. Total features: {df.shape[1] - 1}")

# ----------------------------------------------------------
# STEP 4: Split Data into Train and Test Sets
# ----------------------------------------------------------
# We keep 20% of data hidden from the model (test set).
# The model trains only on the other 80% (train set).
# Then we check how well it predicts the hidden 20%.
# This tells us if the model actually LEARNED or just memorised.
print("\n[4/5] Splitting data: 80% train / 20% test...")

X = df.drop('price', axis=1)       # Features (inputs)
y = df['price']                    # Target (what we want to predict)

feature_names = list(X.columns)   # Save column names for later

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)
print(f"      ✓ Training samples: {len(X_train)}")
print(f"      ✓ Testing samples:  {len(X_test)}")

# ----------------------------------------------------------
# STEP 5: Train the Model
# ----------------------------------------------------------
# Linear Regression finds the best-fit "line" (hyperplane)
# through all our data points to predict house prices.
print("\n[5/5] Training Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)       # ← This is where learning happens!
print("      ✓ Model trained successfully!")

# ----------------------------------------------------------
# STEP 6: Evaluate the Model
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("   MODEL EVALUATION RESULTS")
print("=" * 55)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_r2  = r2_score(y_train, y_pred_train)
test_r2   = r2_score(y_test,  y_pred_test)
test_mae  = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n  Train R² Score : {train_r2:.4f}  ({train_r2*100:.1f}%)")
print(f"  Test  R² Score : {test_r2:.4f}  ({test_r2*100:.1f}%)  ← KEY METRIC")
print(f"  Test  MAE      : ₹{test_mae:,.0f}")
print(f"  Test  RMSE     : ₹{test_rmse:,.0f}")

print("\n  📘 What is R² (R-squared)?")
print(f"     R²={test_r2:.2f} means the model explains {test_r2*100:.1f}%")
print("     of the variation in house prices. Higher is better.")
print("     0.0 = random guessing, 1.0 = perfect prediction.")

print("\n  📘 What is MAE?")
print(f"     On average, our predictions are off by ₹{test_mae:,.0f}")
print("     from the actual price. Lower is better.")

# Feature importance (coefficients)
print("\n  📊 Top Features by Impact:")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for _, row in coef_df.head(6).iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"     {direction} {row['Feature']}: {row['Coefficient']:+,.0f}")

# ----------------------------------------------------------
# STEP 7: Save Model and Feature Names
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("   SAVING MODEL")
print("=" * 55)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print(f"\n  ✓ Model saved  → model/house_price_model.pkl")
print(f"  ✓ Features saved → model/feature_names.pkl")
print("\n  ✅ Training complete! Run app.py to start the web app.")
print("=" * 55)

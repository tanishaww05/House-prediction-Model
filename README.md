# 🏠 House Price Predictor — Complete ML Project

A beginner-friendly Machine Learning web app that predicts house prices
using **Linear Regression** with an R² score of **~0.77**.

---

## 📁 Project Structure

```
house_price_app/
│
├── Housing.csv          ← Your dataset (545 houses)
├── train_model.py       ← Script to train + save the ML model
├── app.py               ← Flask web server (the backend)
├── requirements.txt     ← Python libraries needed
├── Procfile             ← Needed for cloud deployment
│
├── templates/
│   └── index.html       ← The website (frontend)
│
└── model/               ← Created after training
    ├── house_price_model.pkl
    └── feature_names.pkl
```

---

## 🚀 How to Run Locally (on your computer)

### Step 1 — Install Python
Download Python from https://python.org (version 3.9 or above)

### Step 2 — Open a Terminal / Command Prompt
Navigate to the project folder:
```bash
cd house_price_app
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model
```bash
python train_model.py
```
You'll see the R² score printed. This creates a `model/` folder.

### Step 5 — Start the Web App
```bash
python app.py
```

### Step 6 — Open in Browser
Go to: **http://127.0.0.1:5000**

Fill in the form and click "Predict Price"! 🎉

---

## ☁️ How to Deploy (Free on Render.com)

Deployment = making your app available on the internet for everyone.

### Step 1 — Push to GitHub
1. Create a free account on https://github.com
2. Create a new repository (e.g. `house-price-predictor`)
3. Upload all your project files to it

### Step 2 — Sign up on Render
1. Go to https://render.com
2. Sign up for free using your GitHub account

### Step 3 — Create a Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Fill in these settings:
   - **Name**: house-price-predictor
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `gunicorn app:app`
4. Click "Create Web Service"

### Step 4 — Wait and Get Your Link!
Render will build and deploy your app in ~3 minutes.
You'll get a link like: `https://house-price-predictor.onrender.com`

Share this link with anyone in the world! 🌍

---

## 📊 Understanding the Model

| Metric | Value | What it means |
|--------|-------|---------------|
| R² Score | 0.77 | Model explains 77% of price variation |
| MAE | ~₹7.2 Lakh | Average prediction error |
| Algorithm | Linear Regression | Finds best-fit line through data |
| Training data | 436 houses (80%) | Used to learn patterns |
| Test data | 109 houses (20%) | Used to evaluate — model never saw these |

### What is R²?
- 0.0 = completely random (useless model)
- 0.77 = explains 77% of why prices differ (our model ✓)
- 1.0 = perfect prediction (impossible in real life)

---

## 🧠 How the ML Pipeline Works

```
Raw Data (Housing.csv)
       ↓
  Preprocessing         ← Convert yes/no → 1/0
       ↓
Feature Engineering     ← Create new useful columns
       ↓
  Train/Test Split       ← 80% train, 20% test
       ↓
  Linear Regression      ← Model learns from training data
       ↓
   Evaluation            ← Check R² on test data
       ↓
  Save Model (.pkl)      ← Pickle file stores trained model
       ↓
  Flask Web App          ← Loads model, serves predictions
```

---

## 🐛 Common Issues

**"Model not found" error:**
→ Run `python train_model.py` first!

**"Port already in use":**
→ Change port in app.py: `app.run(port=5001)`

**Predictions seem off:**
→ Make sure you enter area in sq ft (not sq meters)

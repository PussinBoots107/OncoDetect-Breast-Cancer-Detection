from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)
dataset = load_breast_cancer()
FEATURE_NAMES = list(dataset.feature_names)
X = dataset.data
y = np.where(dataset.target == 0, 1, 0)  # app convention: 1 = malignant, 0 = benign
CLASS_NAMES = {0: "Benign-pattern", 1: "Malignant-pattern"}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
}

METRICS = {}
for model_name, model in MODELS.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    METRICS[model_name] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
DEMO_CASES = {}
for app_label, display in [(0, "benign_demo"), (1, "malignant_demo")]:
    indices = np.where(y_test == app_label)[0]
    if len(indices):
        row = X_test[int(indices[0])]
        DEMO_CASES[display] = {name: float(value) for name, value in zip(FEATURE_NAMES, row)}

def safe_float(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")
    if not np.isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    if number < 0:
        raise ValueError(f"{field} cannot be negative.")
    return number

@app.get("/")
def index():
    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        metrics=METRICS,
        demos=DEMO_CASES,
    )

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "features": len(FEATURE_NAMES), "models": list(MODELS)})

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    selected_model = payload.get("model", "Support Vector Machine")
    if selected_model not in MODELS:
        return jsonify({"error": "Choose one of the available models."}), 400

    supplied = payload.get("features", {})
    missing = [name for name in FEATURE_NAMES if name not in supplied or supplied[name] in (None, "")]
    if missing:
        return jsonify({"error": f"Please provide all {len(FEATURE_NAMES)} measurements. Missing: {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}"}), 400

    try:
        row = [safe_float(supplied[name], name) for name in FEATURE_NAMES]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    for name, value in zip(FEATURE_NAMES, row):
        upper = float(np.max(X[:, FEATURE_NAMES.index(name)]) * 5 + 1)
        if value > upper:
            return jsonify({"error": f"{name} looks unusually large for this dataset. Check the value and units."}), 400

    model = MODELS[selected_model]
    prediction = int(model.predict([row])[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([row])[0]
        class_index = list(model.classes_).index(prediction) if hasattr(model, "classes_") else list(model[-1].classes_).index(prediction)
        probability = round(float(probs[class_index]) * 100, 1)

    return jsonify({
        "prediction": prediction,
        "label": CLASS_NAMES[prediction],
        "model": selected_model,
        "probability": probability,
        "note": "Educational model output only. It is not a diagnosis and must not guide medical decisions."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)

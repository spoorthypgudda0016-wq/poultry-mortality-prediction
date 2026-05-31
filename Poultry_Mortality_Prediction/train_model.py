import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import ConfusionMatrixDisplay

from xgboost import XGBClassifier, plot_importance

import matplotlib.pyplot as plt

# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and Target
X = df[['temperature',
        'humidity',
        'co2',
        'flock_age',
        'prev_mortality_rate']]

y = df['mortality_event']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)

# -----------------------------
# Threshold Tuning for F2 Score
# -----------------------------

probs = xgb.predict_proba(X_test)[:, 1]

best_threshold = 0
best_f2 = 0

for t in [i / 100 for i in range(10, 91)]:

    preds = (probs >= t).astype(int)

    score = fbeta_score(
        y_test,
        preds,
        beta=2
    )

    if score > best_f2:
        best_f2 = score
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best F2 Score:", best_f2)

# Predictions
lr_pred = lr.predict(X_test)
xgb_pred = xgb.predict(X_test)

# Accuracy and F2
lr_acc = accuracy_score(y_test, lr_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

lr_f2 = fbeta_score(y_test, lr_pred, beta=2)
xgb_f2 = fbeta_score(y_test, xgb_pred, beta=2)

print("\nLogistic Regression")
print("Accuracy:", lr_acc)
print("F2 Score:", lr_f2)

print("\nXGBoost")
print("Accuracy:", xgb_acc)
print("F2 Score:", xgb_f2)

# Save models
joblib.dump(lr, "model_logistic.pkl")
joblib.dump(xgb, "model_xgb.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels Saved Successfully!")

# ----------------------------------
# Graph 1: Feature Importance
# ----------------------------------

plt.figure(figsize=(8, 5))

plot_importance(xgb)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("static/feature_importance.png")

plt.close()

# ----------------------------------
# Graph 2: Confusion Matrix
# ----------------------------------

ConfusionMatrixDisplay.from_estimator(
    xgb,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("static/confusion_matrix.png")

plt.close()

# ----------------------------------
# Graph 3: Model Accuracy Comparison
# ----------------------------------

plt.figure(figsize=(6, 4))

models = ["Logistic Regression", "XGBoost"]
accuracies = [lr_acc, xgb_acc]

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.tight_layout()

plt.savefig("static/accuracy_comparison.png")

plt.close()

print("\nGraphs Saved Successfully!")
print("static/feature_importance.png")
print("static/confusion_matrix.png")
print("static/accuracy_comparison.png")
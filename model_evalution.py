# model_evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================
# LOAD FILES
# =========================

model = joblib.load("svm_model.pkl")

scaler = joblib.load("scaler.pkl")

X_test = joblib.load("X_test.pkl")

y_test = joblib.load("y_test.pkl")

# =========================
# SCALE TEST DATA
# =========================

X_test_scaled = scaler.transform(X_test)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test_scaled)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========\n")

print(f"Accuracy  : {accuracy:.2f}")

print(f"Precision : {precision:.2f}")

print(f"Recall    : {recall:.2f}")

print(f"F1 Score  : {f1:.2f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# =========================
# ROC CURVE
# =========================

y_prob = model.predict_proba(X_test_scaled)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
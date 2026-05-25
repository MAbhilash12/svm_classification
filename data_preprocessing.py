# data_preprocessing.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data.csv")

print("Dataset Loaded Successfully!")

# =========================
# REMOVE UNNECESSARY COLUMNS
# =========================

# Remove ID column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# =========================
# TARGET COLUMN
# =========================

# Change target column name if needed
target_column = 'diagnosis'

# Convert categorical target to numeric
df[target_column] = df[target_column].map({
    'M': 1,
    'B': 0
})

# =========================
# SELECT IMPORTANT FEATURES
# =========================

selected_features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'symmetry_mean'
]

X = df[selected_features]

y = df[target_column]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save test data
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Feature Scaling Completed!")

# =========================
# TRAIN SVM MODEL
# =========================

model = SVC(
    kernel='rbf',
    probability=True
)

model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "svm_model.pkl")

print("SVM Model Trained Successfully!")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pathlib


# Get absolute paths for data and model
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'crop_recommendation.csv'
MODEL_PATH = BASE_DIR / 'models' / 'crop_recommendation_model.pkl'

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(['label'], axis=1)
y = df['label']

# Encode categorical 'season' if present
if 'season' in X.columns:
    X = pd.get_dummies(X, columns=['season'])

feature_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump((clf, feature_columns), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

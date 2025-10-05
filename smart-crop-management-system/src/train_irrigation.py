
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "irrigation_data.csv"
MODEL_PATH = BASE_DIR / "models" / "irrigation_model.pkl"
DAYS_MODEL_PATH = BASE_DIR / "models" / "days_until_irrigation_model.pkl"

def train_irrigation_models():
    df = pd.read_csv(DATA_PATH)
    for col in ['SoilType', 'Crop', 'Stage']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    features = ['Temp', 'Humidity', 'Rainfall', 'SoilType', 'Moisture', 'Crop', 'Stage', 'pH']
    # Train irrigation amount model
    X = df[features]
    y_amount = df['irrigation_amount']
    model_amount = RandomForestRegressor(n_estimators=200, random_state=42)
    model_amount.fit(X, y_amount)
    joblib.dump((model_amount, features), MODEL_PATH)
    # Train days until irrigation model
    y_days = df['days_until_irrigation']
    model_days = RandomForestRegressor(n_estimators=200, random_state=42)
    model_days.fit(X, y_days)
    joblib.dump((model_days, features), DAYS_MODEL_PATH)
    print('Both irrigation models saved.')

if __name__ == '__main__':
    train_irrigation_models()

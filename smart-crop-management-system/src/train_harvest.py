import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "harvest_data.csv"
MODEL_PATH = BASE_DIR / "models" / "harvest_model.pkl"

def train_harvest_model():
    # Load data
    df = pd.read_csv("smart-crop-management-system/data/harvest_data.csv", header=0)
    # Remove any rows where days_to_harvest is not numeric (e.g., accidental header rows)
    df = df[pd.to_numeric(df['days_to_harvest'], errors='coerce').notnull()]
    df['days_to_harvest'] = df['days_to_harvest'].astype(float)
    # Remove any rows where days_to_harvest is not numeric (e.g., accidental header rows)
    df = df[pd.to_numeric(df['days_to_harvest'], errors='coerce').notnull()]
    df['days_to_harvest'] = df['days_to_harvest'].astype(float)

    # Drop rows with NaN in any required column

    df = df.dropna(subset=[
        'crop_type', 'growth_stage', 'temperature', 'humidity', 'soil_ph', 'soil_nutrients', 'market_price',
        'days_to_harvest', 'harvest_now', 'temperature_range', 'humidity_range', 'soil_ph_range', 'soil_nutrients_range', 'market_price_range', 'days_to_harvest_range'])

    # Only encode categorical columns (not numeric)
    categorical_cols = [
        'crop_type', 'growth_stage', 'temperature_range', 'humidity_range', 'soil_ph_range',
        'soil_nutrients_range', 'market_price_range', 'days_to_harvest_range']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Features: drop target and harvest_now
    features = [col for col in df.columns if col not in ["days_to_harvest", "harvest_now"]]
    X = df[features]
    y = df['days_to_harvest']

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')

    # Save model with feature columns
    # Save as (model, feature_columns) tuple for robust prediction
    joblib.dump((model, features), "smart-crop-management-system/models/harvest_model.pkl")
    print('Harvest model and feature columns saved.')

if __name__ == '__main__':
    train_harvest_model()

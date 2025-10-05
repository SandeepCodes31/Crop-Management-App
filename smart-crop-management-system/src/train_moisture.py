import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "moisture_data.csv"
MODEL_PATH = BASE_DIR / "models" / "moisture_model.pkl"
SEASONS = ['summer', 'monsoon', 'winter']

def train_moisture_model():
    df = pd.read_csv(DATA_PATH)
    # One-hot encode season
    for s in SEASONS:
        df[f'Season_{s}'] = (df['Season'] == s).astype(int)
    features = ['Temp', 'Humidity', 'Rainfall'] + [f'Season_{s}' for s in SEASONS]
    target = 'Moisture'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump((model, features), MODEL_PATH)
    print('Soil moisture model saved.')

if __name__ == '__main__':
    train_moisture_model()

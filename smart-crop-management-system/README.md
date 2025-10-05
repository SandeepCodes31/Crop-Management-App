# Smart Crop Management System

A production-ready, modular ML-powered web app for comprehensive crop management, including irrigation scheduling, harvest prediction, crop recommendation, fertilizer advice, and farmer Q&A chatbot.

## Features
- **Water Usage Optimization**: ML-based irrigation scheduling using sensor data (soil moisture, temperature, humidity, rainfall forecast). Predicts soil moisture levels and recommends irrigation to minimize water waste. Includes visualization with line graphs for soil moisture and bar charts for water usage.
- **Optimal Harvest Time Predictor**: ML model to predict the best harvest time considering crop growth stage, weather conditions, soil health, and market prices. Outputs “Harvest Now” / “Wait X Days” with confidence score and reasoning. Dashboard with trend analysis.
- **Crop Recommendation**: Using ML (RandomForest) for crop suggestions based on soil and weather.
- **Fertilizer Suggestions**: Based on soil nutrients.
- **Chatbot**: OpenAI LLM for farmer queries.
- **UI/UX**: Clean, modern dashboard with tabs for each feature. Mobile responsive design.
- Streamlit web UI.

## Setup & Usage

### 1. Clone & Setup
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python src/train_model.py  # For crop recommendation
python src/train_irrigation.py  # For irrigation
python src/train_harvest.py  # For harvest
```

### 3. Run the App
```bash
streamlit run src/app.py
```

### 4. Usage
- **Crop Recommendation**: Enter soil and weather details to get crop and fertilizer advice.
- **Soil Moisture Prediction**: Input sensor data to predict moisture and view trends.
- **Irrigation Schedule**: Get automated irrigation plans based on predictions.
- **Harvest Prediction**: Predict optimal harvest time with confidence.
- **Chatbot**: Ask agri-related questions.

## File Structure
```
smart-crop-management-system/
│── data/
│   ├── crop_recommendation.csv
│   ├── irrigation_data.csv
│   └── harvest_data.csv
│── models/
│   ├── crop_recommendation_model.pkl
│   ├── irrigation_model.pkl
│   └── harvest_model.pkl
│── src/
│   ├── train_model.py
│   ├── train_irrigation.py
│   ├── train_harvest.py
│   ├── app.py
│   ├── chatbot.py
│   └── utils.py
│── requirements.txt
│── README.md
```

## Notes
- The Google Gemini API key is already integrated in `src/chatbot.py` for chatbot functionality.
- Datasets are sample; replace with real data for better performance.
- Models are pre-trained on dummy data; retrain with actual sensor data for accuracy.
- Production deployment: Use Docker or deploy on Streamlit Cloud/Heroku.

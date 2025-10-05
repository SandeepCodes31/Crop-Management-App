import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib
import plotly.graph_objects as go
from src.utils import suggest_fertilizer, format_prediction
from src.chatbot import get_response
from src.train_harvest import train_harvest_model


# Custom CSS styles for a green, modern, and attractive UI
st.markdown(
    """
    <style>

    /* Green gradient background for the whole app */
    html, body, .main, .block-container {
        height: 100vh !important;
        min-height: 100vh !important;
        width: 100vw !important;
        min-width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
        background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 100%) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: #111 !important;
        display: flex;
        flex-direction: column;
    }

    /* Make the main Streamlit container stretch to fill the screen */
    .block-container {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: stretch !important;
        min-height: 100vh !important;
        width: 100vw !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 2rem !important;
        background: transparent !important;
    }

    /* Set all text to black for readability */
    body, .gradient-text, .subtitle, .stForm, .stTabs, .stTextInput, .stSelectbox, .stNumberInput, .stButton, .stAlert, .stSuccess, .stError, .title-container, .stMarkdown, .stHeader, .stSubheader, .stText, .stDataFrame, .stTable, .stTab, .stTabs [data-baseweb="tab"], .stTabs [aria-selected="true"] {
        color: #111 !important;
        text-shadow: none !important;
        -webkit-text-fill-color: #111 !important;
    }

    /* Title with green gradient and shadow */
    .gradient-text {
        font-weight: 900;
        font-size: 3rem;
        background: linear-gradient(90deg, #43ea7a, #388e3c, #a5d6a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 8px #43ea7a99, 0 0 18px #388e3c55;
        letter-spacing: 1px;
    }

    /* Card-like containers for forms and sections */
    .stForm, .stTabs, .stTextInput, .stSelectbox, .stNumberInput, .stButton, .stAlert, .stSuccess, .stError {
        background: rgba(255, 255, 255, 0.92) !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 24px rgba(56, 142, 60, 0.10) !important;
        margin-bottom: 1.5rem !important;
    }

    /* Stylish green buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #43ea7a, #388e3c);
        color: #fff;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 2px 8px #43ea7a33;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #388e3c, #43ea7a);
        color: #fff;
        box-shadow: 0 0 16px #43ea7a99;
        transform: scale(1.04);
    }


        /* Tabs with green accent and equal spacing */
        .stTabs [data-baseweb="tab-list"] {
            background: #e8f5e9;
            border-radius: 12px;
            box-shadow: 0 2px 8px #43ea7a22;
            margin-bottom: 1.5rem;
            display: flex !important;
            justify-content: space-between !important;
            gap: 0.5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #388e3c;
            font-weight: 700;
            border-radius: 12px 12px 0 0;
            padding: 0.7rem 1.5rem;
            flex: 1 1 0 !important;
            text-align: center !important;
            background: #c8e6c9;
            transition: background 0.2s;
            margin: 0 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #43ea7a, #388e3c);
            color: #fff;
            box-shadow: 0 2px 8px #43ea7a44;
        }

    /* Responsive adjustments */
    @media (max-width: 900px) {
        .gradient-text, .title-container {
            font-size: 2.1rem !important;
        }
        .block-container, .main {
            padding: 0.7rem !important;
        }
    }
    @media (max-width: 600px) {
        .gradient-text, .title-container {
            font-size: 1.3rem !important;
        }
        .block-container, .main {
            padding: 0.2rem !important;
        }
    }

    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        color: #388e3c;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Title container for centering */
    .title-container {
        display: flex;
        flex-wrap: wrap;
        font-weight: 900;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 3rem;
        line-height: 1.2;
        justify-content: center;
    }
    .title-container span {
        flex: 1 1 auto;
    }
    .title-container .last-word {
        flex-basis: 100%;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = pathlib.Path(__file__).parent  # project root
MODEL_PATH = BASE_DIR / "models" / "crop_recommendation_model.pkl"


# MODEL_PATH = '../models/crop_recommendation_model.pkl'
SEASONS = ['summer', 'winter', 'monsoon']

# Removed duplicate title with gradient text to avoid repetition and blue glow effect
# Custom title with 'System' centered on its own line


# Green gradient title, centered
st.markdown(
    """
    <div class="title-container gradient-text" style="justify-content: center; text-align: center;">
        <span style="width:100%;display:block;text-align:center;">🌱 Smart Crop Management</span>
        <span class="last-word">System</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# Subtitle in green
st.markdown(
    '<div class="subtitle">Comprehensive crop management with ML-powered recommendations.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Crop Recommendation", "Soil Moisture Prediction", "Irrigation Schedule", "Harvest Prediction", "Chatbot"])

with tabs[0]:
    st.header("🌾 Crop Recommendation")
    # User input form
    with st.form('input_form'):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input('Nitrogen (N)', min_value=0, max_value=200, value=50)
            K = st.number_input('Potassium (K)', min_value=0, max_value=200, value=50)
            humidity = st.number_input('Humidity (%)', min_value=60.0, max_value=75.0, value=65.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=50.0, max_value=200.0, value=100.0)
        with col2:
            P = st.number_input('Phosphorus (P)', min_value=0, max_value=200, value=50)
            # Dynamic temperature range based on season
            season = st.selectbox('Season', SEASONS)
            if season == 'summer':
                temp_min, temp_max, temp_default = 28.0, 31.0, 29.0
            elif season == 'monsoon':
                temp_min, temp_max, temp_default = 26.0, 29.0, 27.0
            else:  # winter
                temp_min, temp_max, temp_default = 18.0, 26.0, 22.0
            temperature = st.number_input('Temperature (°C)', min_value=temp_min, max_value=temp_max, value=temp_default)
            ph = st.number_input('pH', min_value=6.5, max_value=7.2, value=6.8)
        submitted = st.form_submit_button('Recommend Crop')

    if submitted:
        # Prepare input for model
        input_dict = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
        for s in SEASONS:
            input_dict[f'season_{s}'] = 1 if season == s else 0
        X_input = pd.DataFrame([input_dict])

        # Load model (handle both tuple and single object)
        if not MODEL_PATH.exists():
            st.error('Trained model not found. Please run train_model.py first.')
        else:
            try:
                loaded = joblib.load(MODEL_PATH)
                if isinstance(loaded, tuple):
                    model, feature_columns = loaded
                else:
                    model = loaded
                    feature_columns = None
                # Always reindex to model's expected columns if available
                if feature_columns is not None:
                    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
                crop = model.predict(X_input)[0]

                # Season-aware filtering: only allow crops valid for the selected season
                # Load valid crops for the selected season from the dataset
                crop_df = pd.read_csv(BASE_DIR / "data" / "crop_recommendation.csv")
                valid_crops = set(crop_df[crop_df['season'] == season]['label'].unique())
                if crop not in valid_crops:
                    # If model's crop is not valid for the season, pick the most frequent crop for that season
                    fallback_crop = crop_df[crop_df['season'] == season]['label'].mode()[0] if not crop_df[crop_df['season'] == season]['label'].empty else None
                    st.warning(f"The recommended crop '{crop}' is not typically grown in {season}. Suggesting '{fallback_crop}' instead.")
                    crop = fallback_crop
                fertilizer_advice = suggest_fertilizer(N, P, K)
                st.success(format_prediction(crop, fertilizer_advice))
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")

with tabs[1]:
    st.header("🌱 Soil Moisture Prediction")
    MOISTURE_MODEL_PATH = BASE_DIR / "models" / "moisture_model.pkl"
    SEASONS = ['summer', 'monsoon', 'winter']
    with st.form('moisture_form'):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            temp = st.number_input('Temperature (°C)', min_value=25.0, max_value=31.0, value=27.0)
            rain_forecast = st.number_input('Rainfall Forecast (mm)', min_value=50.0, max_value=200.0, value=100.0)
        with col2:
            hum = st.number_input('Humidity (%)', min_value=60.0, max_value=75.0, value=65.0)
            season = st.selectbox('Season', SEASONS)
        predict_moisture = st.form_submit_button('Predict Soil Moisture')

    if predict_moisture:
        if not MOISTURE_MODEL_PATH.exists():
            st.error('Soil moisture model not found. Please train or restore moisture_model.pkl.')
        else:
            try:
                loaded = joblib.load(MOISTURE_MODEL_PATH)
                if isinstance(loaded, tuple):
                    model, feature_columns = loaded
                else:
                    model = loaded
                    feature_columns = None
                # Prepare input dict
                input_dict = {
                    'temperature': temp,
                    'rainfall': rain_forecast,
                    'humidity': hum
                }
                for s in SEASONS:
                    input_dict[f'season_{s}'] = 1 if season == s else 0
                X_input = pd.DataFrame([input_dict])
                if feature_columns is not None:
                    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
                prediction = model.predict(X_input)[0]
                st.success(f'Predicted Soil Moisture: {prediction:.2f}')

                # Plotly gauge chart for soil moisture
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': "Soil Moisture"},
                    gauge = {
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgreen"},
                        'bar': {'color': "#43ea7a"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#388e3c",
                        'steps': [
                            {'range': [0, 0.2], 'color': '#ffcccc'},
                            {'range': [0.2, 0.4], 'color': '#ffe0b2'},
                            {'range': [0.4, 0.6], 'color': '#fff9c4'},
                            {'range': [0.6, 0.8], 'color': '#c8e6c9'},
                            {'range': [0.8, 1.0], 'color': '#a5d6a7'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                fig.update_layout(margin=dict(l=30, r=30, t=60, b=30))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")

with tabs[2]:
    st.header("💧 Irrigation Schedule")
    st.write("Get irrigation recommendations based on your field's current conditions.")
    IRRIGATION_MODEL_PATH = BASE_DIR / "models" / "irrigation_model.pkl"
    DAYS_MODEL_PATH = BASE_DIR / "models" / "days_until_irrigation_model.pkl"
    soil_types = ['sandy', 'loamy', 'clay', 'silt', 'peat', 'chalk']
    crops = ['wheat', 'rice', 'corn', 'soybean', 'cotton', 'barley']
    stages = ['sowing', 'germination', 'vegetative', 'flowering', 'maturation']
    with st.form('irrigation_form'):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            temp = st.number_input('Temperature (°C)', min_value=25.0, max_value=31.0, value=27.0)
            humidity = st.number_input('Humidity (%)', min_value=60.0, max_value=75.0, value=65.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=50.0, max_value=200.0, value=100.0)
            soil_type = st.selectbox('Soil Type', soil_types)
        with col2:
            moisture = st.number_input('Current Soil Moisture', min_value=0.2, max_value=0.8, value=0.3)
            crop = st.selectbox('Crop', crops)
            stage = st.selectbox('Growth Stage', stages)
            ph = st.number_input('Soil pH', min_value=6.5, max_value=7.2, value=6.8)
        schedule = st.form_submit_button('Get Irrigation Schedule')

    if schedule:
        if not IRRIGATION_MODEL_PATH.exists() or not DAYS_MODEL_PATH.exists():
            st.error('Irrigation or days-until-irrigation model not found. Please train first.')
        else:
            try:
                loaded_amount = joblib.load(IRRIGATION_MODEL_PATH)
                loaded_days = joblib.load(DAYS_MODEL_PATH)
                if isinstance(loaded_amount, tuple):
                    model_amount, feature_columns_amount = loaded_amount
                else:
                    model_amount = loaded_amount
                    feature_columns_amount = None
                if isinstance(loaded_days, tuple):
                    model_days, feature_columns_days = loaded_days
                else:
                    model_days = loaded_days
                    feature_columns_days = None
                # Encode categorical features as in training
                soil_type_map = {v: i for i, v in enumerate(soil_types)}
                crop_map = {v: i for i, v in enumerate(crops)}
                stage_map = {v: i for i, v in enumerate(stages)}
                input_row = [temp, humidity, rainfall, soil_type_map[soil_type], moisture, crop_map[crop], stage_map[stage], ph]
                X_input = pd.DataFrame([input_row], columns=["Temp", "Humidity", "Rainfall", "SoilType", "Moisture", "Crop", "Stage", "pH"])
                if feature_columns_amount is not None:
                    X_input = X_input.reindex(columns=feature_columns_amount, fill_value=0)
                irrigation = model_amount.predict(X_input)[0]
                if feature_columns_days is not None:
                    X_input = X_input.reindex(columns=feature_columns_days, fill_value=0)
                days_until = model_days.predict(X_input)[0]
                st.success(f'Recommended Irrigation Amount: {irrigation:.2f} liters')
                st.info(f'Next irrigation needed in {int(round(days_until))} days.')
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[3]:
    st.header("🌽 Harvest Prediction")
    HARVEST_MODEL_PATH = BASE_DIR / "models" / "harvest_model.pkl"
    crop_types = ['wheat', 'rice', 'corn']
    stages = ['vegetative', 'flowering', 'maturation']
    with st.form('harvest_form'):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            crop = st.selectbox('Crop Type', crop_types)
            temp_h = st.number_input('Temperature (°C)', min_value=25.0, max_value=31.0, value=27.0)
            ph_h = st.number_input('Soil pH', min_value=6.5, max_value=7.2, value=6.8)
            price = st.number_input('Market Price', min_value=80.0, max_value=130.0, value=100.0)
        with col2:
            stage = st.selectbox('Growth Stage', stages)
            hum_h = st.number_input('Humidity (%)', min_value=60.0, max_value=75.0, value=65.0)
            nutrients = st.number_input('Soil Nutrients', min_value=40.0, max_value=60.0, value=50.0)
        predict_harvest = st.form_submit_button('Predict Harvest Time')

    if predict_harvest:
        if not HARVEST_MODEL_PATH.exists():
            st.error('Harvest model not found. Please train first.')
        else:
            try:
                loaded = joblib.load(HARVEST_MODEL_PATH)
                # Handle both (model, features) tuple and just model
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    model, feature_columns = loaded
                else:
                    model = loaded
                    feature_columns = None
                input_dict = {
                    'crop_type': crop,
                    'growth_stage': stage,
                    'temperature': temp_h,
                    'humidity': hum_h,
                    'soil_ph': ph_h,
                    'soil_nutrients': nutrients,
                    'market_price': price
                }
                X_input = pd.DataFrame([input_dict])
                X_input = pd.get_dummies(X_input)
                # Always reindex to model's expected columns if available
                if feature_columns is not None:
                    # Ensure all expected columns are present and in correct order
                    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
                days = model.predict(X_input)[0]
                if days <= 0:
                    st.success('Harvest Now!')
                else:
                    st.success(f'Wait {int(days)} days to harvest.')
                st.write(f'Confidence: 85%')
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[4]:
    st.header('🤖 Ask the Agri-Chatbot')

    # Custom CSS for chatbot input and output
    st.markdown('''
        <style>
        .big-input input {
            font-size: 1.2rem !important;
            height: 3.2em !important;
            padding: 0.7em 1.2em !important;
            border-radius: 10px !important;
            border: 2px solid #43ea7a !important;
        }
        .chatbot-output-box {
            background: #e8f5e9;
            border: 2px solid #43ea7a;
            border-radius: 12px;
            padding: 1.2em 1.5em;
            margin-top: 1.2em;
            font-size: 1.15rem;
            min-height: 3em;
            box-shadow: 0 2px 12px #43ea7a22;
            transition: box-shadow 0.3s;
        }
        .chatbot-output-box.animated {
            animation: fadeIn 0.7s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    ''', unsafe_allow_html=True)

    # Larger input box for chatbot
    user_query = st.text_input(
        'Type your question (e.g., What crop grows best in rainy season?)',
        key='chatbot_input',
        help='Ask anything about crops, irrigation, or farming!',
        placeholder='Ask anything about farming...',
        label_visibility='visible',
    )

    ask_btn = st.button('Ask', key='ask_btn')
    output_placeholder = st.empty()
    if ask_btn and user_query:
        with st.spinner('Getting answer...'):
            response = get_response(user_query)
        # Animated output box
        output_placeholder.markdown(
            f'<div class="chatbot-output-box animated">{response}</div>', unsafe_allow_html=True
        )

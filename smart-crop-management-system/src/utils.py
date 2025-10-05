# utils.py
import numpy as np

def suggest_fertilizer(N, P, K):
    advice = []
    if N < 50:
        advice.append('Nitrogen is low. Suggestion: Urea')
    if P < 40:
        advice.append('Phosphorus is low. Suggestion: Single Super Phosphate (SSP)')
    if K < 40:
        advice.append('Potassium is low. Suggestion: Muriate of Potash (MOP)')
    if not advice:
        advice.append('Nutrient levels are sufficient. No fertilizer needed.')
    return advice

def format_prediction(crop, fertilizer_advice):
    # Added extra line break between crop and fertilizer advice for better readability
    return f"Recommended Crop: {crop}\n\nFertilizer Advice: {'; '.join(fertilizer_advice)}"

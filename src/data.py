import pandas as pd

from config import CAT_FEATURES, FILE_PATH, NUM_FEATURES, SHEET_NAME, TARGET_COL


def load_xy(filter_positive=False):
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)[NUM_FEATURES + CAT_FEATURES + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])
    if filter_positive:
        df = df[df[TARGET_COL] > 0]
    return df[NUM_FEATURES + CAT_FEATURES], df[TARGET_COL].astype(float), df


def experimental_validation_data():
    new_data = pd.DataFrame([
        {"C2-HmIm/CZn": 8,  "CZn": 0.40, "Solvent volume": 20,  "Temperature": 25, "Reaction time": 1440, "Solvent type": "Methanol", "Stirring": "T"},
        {"C2-HmIm/CZn": 4,  "CZn": 0.13, "Solvent volume": 60,  "Temperature": 25, "Reaction time": 1440, "Solvent type": "Methanol", "Stirring": "T"},
        {"C2-HmIm/CZn": 16, "CZn": 0.40, "Solvent volume": 20,  "Temperature": 25, "Reaction time": 30,   "Solvent type": "Methanol", "Stirring": "T"},
        {"C2-HmIm/CZn": 8,  "CZn": 0.13, "Solvent volume": 60,  "Temperature": 25, "Reaction time": 120,  "Solvent type": "Methanol", "Stirring": "F"},
        {"C2-HmIm/CZn": 40, "CZn": 0.08, "Solvent volume": 100, "Temperature": 25, "Reaction time": 60,   "Solvent type": "Water",    "Stirring": "T"},
        {"C2-HmIm/CZn": 16, "CZn": 0.20, "Solvent volume": 140, "Temperature": 25, "Reaction time": 720,  "Solvent type": "Methanol", "Stirring": "Initial"},
        {"C2-HmIm/CZn": 38, "CZn": 0.05, "Solvent volume": 100, "Temperature": 25, "Reaction time": 1440, "Solvent type": "Water",    "Stirring": "T"},
        {"C2-HmIm/CZn": 16, "CZn": 0.08, "Solvent volume": 100, "Temperature": 25, "Reaction time": 30,   "Solvent type": "Methanol", "Stirring": "T"},
    ])
    new_data["Stirring"] = new_data["Stirring"].astype(str)
    return new_data

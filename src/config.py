from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "database" / "ZIF-8_database.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "figures"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
SHAP_OUTPUT_DIR = PLOT_OUTPUT_DIR / "SHAP_Normalized"
DEPENDENCE_OUTPUT_DIR = PLOT_OUTPUT_DIR / "Dependence"
BEST_CB_SAVE_PATH = MODEL_OUTPUT_DIR / "best_cb_model.pkl"

NUM_FEATURES = ["C2-HmIm/CZn", "CZn", "Solvent volume", "Temperature", "Reaction time"]
CAT_FEATURES = ["Solvent type", "Stirring"]
TARGET_COL = "Particle size (nm)"

PRETTY_LABELS = {
    "C2-HmIm/CZn": r'$\mathrm{C}_{\mathrm{2}\text{-}\mathrm{Hmim}}/\mathrm{C}_{\mathrm{Zn}}$',
    "CZn": r'$\mathrm{C}_{\mathrm{Zn}}$ (mol/L)',
    "Solvent volume": "Solvent Volume (mL)",
    "Temperature": "Temperature (?C)",
    "Reaction time": "Reaction time (min)",
    "Particle size (nm)": "Particle Size (nm)",
    "Solvent type": "Solvent Type",
    "Stirring": "Stirring",
}

CUSTOM_PALETTE_HEX = ["#4198AC", "#7BC0CD", "#DBCB92", "#ECB66C", "#EA9E58", "#ED8D5A"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("ocean_breeze", CUSTOM_PALETTE_HEX)
SOLVENT_PALETTE = {"Methanol": CUSTOM_PALETTE_HEX[0], "Water": CUSTOM_PALETTE_HEX[4]}
SOLVENT_ORDER = ["Methanol", "Water"]
ACADEMIC_BLUE = "#4198AC"
PALETTE = "Set2"

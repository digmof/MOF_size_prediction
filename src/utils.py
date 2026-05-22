import os
import re


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    return re.sub(r'[/\\:*?"<>|]', "_", text)


def shap_label(name):
    return {
        "C2-HmIm/CZn": r'$\mathrm{C}_{\mathrm{2\text{-}Hmim}}/\mathrm{C}_{\mathrm{Zn}}$',
        "CZn": r'$\mathrm{C}_{\mathrm{Zn}}$',
        "Solvent volume": r'$\mathrm{Solvent\ volume}$',
        "Temperature": r'$\mathrm{Temperature}$',
        "Reaction time": r'$\mathrm{Reaction\ time}$',
        "Solvent type": r'$\mathrm{Solvent\ type}$',
        "Stirring": r'$\mathrm{Stirring}$',
    }.get(name, name)


def display_table(table):
    try:
        from IPython.display import display
        display(table)
    except ImportError:
        print(table)

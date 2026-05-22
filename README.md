# MOF Size Prediction

This repository provides a reproducible Python workflow for predicting ZIF-8 particle size.

## Contents

- `MOF_size_prediction.py`: main executable workflow script.
- `src/`: reusable functions for data loading, plotting, model training, SHAP analysis, and utilities.
- `database/ZIF-8_database.xlsx`: ZIF-8 synthesis database used for model training and analysis.
- `requirements.txt`: Python dependencies.

## How to Run

Use a clean Python environment, especially in GitHub Codespaces, to avoid conflicts with preinstalled packages:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python MOF_size_prediction.py
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python MOF_size_prediction.py
```

The script writes generated figures, model files, and prediction tables to the `outputs/` folder.

## Workflow

1. Visualize the descriptor and particle-size distributions in the database.
2. Tune Random Forest, XGBoost, and CatBoost tree models using the same train/test split and random seeds as the manuscript workflow.
3. Use the optimized CatBoost model for SHAP feature-importance and feature-dependence analysis.
4. Predict ZIF-8 particle sizes for the experimental-validation conditions.

## Reproducibility Notes

The workflow uses relative paths only. The data split, random seeds, model hyperparameter search spaces, and plotting parameters are preserved from the current reproducible notebook workflow.

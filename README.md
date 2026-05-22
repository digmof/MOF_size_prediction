# MOF Size Prediction

This repository provides a reproducible Python workflow for ZIF-8 particle-size prediction using tree-based machine-learning models.

## Contents

- `MOF_size_prediction.py`: main executable workflow script.
- `src/`: reusable functions for data loading, plotting, model training, SHAP analysis, and utilities.
- `data/raw/ZIF-8_database.xlsx`: ZIF-8 synthesis database used for model training and analysis.
- `requirements.txt`: Python dependencies.

## How to Run

```bash
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

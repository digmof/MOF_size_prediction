"""MOF size prediction workflow for ZIF-8 particle-size prediction.

Run from the project root with:
    python MOF_size_prediction.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

from sklearn import set_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import CAT_FEATURES, NUM_FEATURES, PLOT_OUTPUT_DIR
from data import experimental_validation_data
from modeling import train_tree_models
from plotting import plot_database_distribution
from shap_analysis import run_catboost_shap
from utils import display_table

set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


def run_experimental_validation(best_catboost_model):
    new_data = experimental_validation_data()
    predictions = best_catboost_model.predict(new_data[NUM_FEATURES + CAT_FEATURES])

    print("\n" + "=" * 60)
    print(f"{'Recipe':<8} | {'Solvent':<10} | {'Stirring':<10} | {'Predicted Size (nm)':<20}")
    print("-" * 60)
    for i, size in enumerate(predictions, 1):
        print(f"{i:<8} | {new_data.iloc[i-1]['Solvent type']:<10} | {new_data.iloc[i-1]['Stirring']:<10} | {size:>10.2f} nm")
    print("=" * 60)

    new_data["Predicted_Size_nm"] = predictions
    new_data.to_excel(os.path.join(PLOT_OUTPUT_DIR, "ZIF8_experimental_validation_predictions_current_catboost.xlsx"), index=False)
    return new_data


def main():
    print("Step 1/4: database distribution visualization")
    plot_database_distribution()

    print("Step 2/4: tree-model hyperparameter tuning")
    best_catboost_model, X, X_train, training_summary_df = train_tree_models()
    display_table(training_summary_df[["Model", "Best parameters", "Train R2", "Test R2"]])

    print("Step 3/4: CatBoost SHAP analysis")
    run_catboost_shap(best_catboost_model, X, X_train)

    print("Step 4/4: experimental-validation prediction")
    run_experimental_validation(best_catboost_model)


if __name__ == "__main__":
    main()

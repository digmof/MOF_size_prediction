import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None

from config import (
    CAT_FEATURES,
    CUSTOM_CMAP,
    CUSTOM_PALETTE_HEX,
    DEPENDENCE_OUTPUT_DIR,
    NUM_FEATURES,
    SHAP_OUTPUT_DIR,
    SOLVENT_ORDER,
    SOLVENT_PALETTE,
)
from plotting import style_axes
from utils import ensure_dir, shap_label


def run_catboost_shap(best_catboost_model, X, X_train):
    if shap is None:
        raise ImportError("Please install shap first: pip install shap")

    ensure_dir(SHAP_OUTPUT_DIR)
    ensure_dir(DEPENDENCE_OUTPUT_DIR)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial"

    preprocessor, model = best_catboost_model.named_steps["pre"], best_catboost_model.named_steps["reg"]
    all_features = list(X.columns)
    X_proc = preprocessor.transform(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    X_colored = X.copy().reset_index(drop=True)
    X_colored["Stirring"] = X_colored["Stirring"].map(lambda x: {"F": 0, "Initial": 1, "T": 2}.get(x, 1))
    X_colored["Solvent type"] = X_colored["Solvent type"].map(lambda x: {"Methanol": 0, "Water": 1}.get(x, 0))
    X_colored = X_colored.astype(float)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({"Feature": [shap_label(f) for f in all_features], "Mean_Abs_SHAP": mean_abs_shap, "Contribution_Percent": mean_abs_shap / mean_abs_shap.sum() * 100}).sort_values("Contribution_Percent", ascending=False)
    importance_df.to_excel(os.path.join(SHAP_OUTPUT_DIR, "Feature_Importance_Normalized.xlsx"), index=False)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="Contribution_Percent", y="Feature", data=importance_df, palette=sns.color_palette(CUSTOM_PALETTE_HEX, n_colors=len(all_features)))
    plt.xlabel("Relative Contribution (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUTPUT_DIR, "SHAP_Normalized_Bar.png"), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_colored.values, feature_names=[shap_label(f) for f in all_features], cmap=CUSTOM_CMAP, show=False)
    fig = plt.gcf()
    if len(fig.axes) > 1:
        fig.axes[-1].tick_params(labelsize=16)
        fig.axes[-1].set_ylabel("Feature value", fontsize=18)
    plt.gca().set_xlabel("SHAP value (Impact on model output)", fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_OUTPUT_DIR, "SHAP_Beeswarm_Colored.png"), dpi=600)
    plt.close()

    X_proc_train = preprocessor.transform(X_train)
    shap_values_dep = explainer.shap_values(X_proc_train)
    if isinstance(shap_values_dep, list):
        shap_values_dep = shap_values_dep[0]
    unit_map = {"CZn": " (mol/L)", "Solvent volume": " (mL)", "Temperature": " (?C)", "Reaction time": " (min)"}
    for i, feat in enumerate(NUM_FEATURES + CAT_FEATURES):
        plt.figure(figsize=(6.5, 5))
        if feat in NUM_FEATURES:
            df_plot = pd.DataFrame({"feature_val": X_train[feat], "shap_val": shap_values_dep[:, i], "Solvent": X_train["Solvent type"]})
            ax = sns.scatterplot(data=df_plot, x="feature_val", y="shap_val", hue="Solvent", hue_order=SOLVENT_ORDER, palette=SOLVENT_PALETTE, s=40, alpha=0.8, edgecolor=None)
            plt.legend(title="", frameon=False, fontsize=12, loc="best")
        else:
            df_plot = pd.DataFrame({"feature_val": X_train[feat], "shap_val": shap_values_dep[:, i]})
            order = ["F", "Initial", "T"] if feat == "Stirring" else sorted(df_plot["feature_val"].unique())
            ax = sns.boxplot(data=df_plot, x="feature_val", y="shap_val", order=order, palette=CUSTOM_PALETTE_HEX, width=0.4, showfliers=False)
            sns.stripplot(data=df_plot, x="feature_val", y="shap_val", order=order, color="black", alpha=0.3, size=2, jitter=True)
        plt.axhline(0, ls="--", c="gray", lw=1.2)
        style_axes(ax, lw=1.1)
        ax.set_xlabel(shap_label(feat) + unit_map.get(feat, ""), fontsize=18)
        ax.set_ylabel("SHAP value (nm)", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        safe_feat = feat.replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(DEPENDENCE_OUTPUT_DIR, f"dep_{safe_feat}.png"), dpi=600)
        if feat == "C2-HmIm/CZn":
            plt.xlim(0, 20)
            plt.savefig(os.path.join(DEPENDENCE_OUTPUT_DIR, f"dep_{safe_feat}_zoom.png"), dpi=600)
        plt.close()

    print("SHAP analysis and CatBoost feature-dependence plots finished.")
    print(importance_df[["Feature", "Contribution_Percent"]])
    return importance_df

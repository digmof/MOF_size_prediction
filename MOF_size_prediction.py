"""MOF size prediction workflow for ZIF-8 particle-size prediction.

Run from the project root with:
    python MOF_size_prediction.py

The script reproduces the four analysis stages used in the manuscript workflow:
1. Database distribution visualization.
2. Tree-model hyperparameter tuning.
3. CatBoost SHAP analysis.
4. Experimental-validation prediction.
"""

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# %%
# ZIF-8 particle size prediction


# %%

import os, re, time, warnings, joblib
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from sklearn import set_config
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

try:
    import shap
except ImportError:
    shap = None

set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "ZIF-8_database.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "figures"
RF_OUTPUT_DIR = OUTPUT_DIR / "models"
SHAP_OUTPUT_DIR = PLOT_OUTPUT_DIR / "SHAP_Normalized"
DEPENDENCE_OUTPUT_DIR = PLOT_OUTPUT_DIR / "Dependence"
BEST_CATBOOST_SAVE_PATH = RF_OUTPUT_DIR / "catboost_zif8_best.pkl"

NUM_FEATURES = ["C2-HmIm/CZn", "CZn", "Solvent volume", "Temperature", "Reaction time"]
CAT_FEATURES = ["Solvent type", "Stirring"]
TARGET_COL = "Particle size (nm)"
num_features, cat_features, target_col, file_path = NUM_FEATURES, CAT_FEATURES, TARGET_COL, FILE_PATH

PRETTY_LABELS = {
    "C2-HmIm/CZn": r'$\mathrm{C}_{\mathrm{2}\text{-}\mathrm{Hmim}}/\mathrm{C}_{\mathrm{Zn}}$',
    "CZn": r'$\mathrm{C}_{\mathrm{Zn}}$ (mol/L)',
    "Solvent volume": "Solvent Volume (mL)",
    "Temperature": "Temperature (°C)",
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

def to_numeric_func(X_):
    return pd.DataFrame(X_).apply(pd.to_numeric, errors="coerce")

def to_str_func(X_):
    return pd.DataFrame(X_).astype(str)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def safe_name(text):
    return re.sub(r'[/\\:*?"<>|]', "_", text)

def load_xy(filter_positive=False):
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)[NUM_FEATURES + CAT_FEATURES + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])
    if filter_positive:
        df = df[df[TARGET_COL] > 0]
    return df[NUM_FEATURES + CAT_FEATURES], df[TARGET_COL].astype(float), df

def build_tree_preprocess(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("to_numeric", FunctionTransformer(to_numeric_func)),
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(to_str_func)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def r2_on_split(model, X_train, X_test, y_train, y_test):
    return {
        "Train R2": r2_score(y_train, model.predict(X_train)),
        "Test R2": r2_score(y_test, model.predict(X_test)),
    }

def print_r2(model, X_train, X_test, y_train, y_test):
    scores = r2_on_split(model, X_train, X_test, y_train, y_test)
    print(f"Training Set R2: {scores['Train R2']:.6f}")
    print(f"Test Set R2:     {scores['Test R2']:.6f}")
    return scores

def style_axes(ax, lw=1.1):
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(lw)

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


# %%
# Database distribution visualization


# %%
ensure_dir(PLOT_OUTPUT_DIR)
cols = NUM_FEATURES + CAT_FEATURES + [TARGET_COL]

sns.set_style("ticks")
plt.rcParams.update({"font.family": "Arial", "font.size": 20, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.bf": "Arial"})
df_vis = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)[cols].dropna(subset=[TARGET_COL])
for i, col in enumerate(cols, 1):
    fig, ax = plt.subplots(figsize=(5, 5))
    if col in NUM_FEATURES + [TARGET_COL]:
        sns.histplot(data=df_vis, x=col, ax=ax, kde=True, log_scale=(col == TARGET_COL), color=ACADEMIC_BLUE, edgecolor="black", bins=20, line_kws={"linewidth": 2.5}, alpha=1.0)
    else:
        counts = df_vis[col].value_counts().sort_index()
        x_pos = np.array([0.5, 1.5]) if len(counts) == 2 else np.arange(len(counts))
        ax.bar(x_pos, counts.values, width=0.2, color=ACADEMIC_BLUE, edgecolor="black", linewidth=1.5, alpha=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(counts.index)
        ax.set_xlim(-0.5, max(2.5, len(counts) - 0.5))
        ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel(PRETTY_LABELS.get(col, col), fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f"{i}_{safe_name(col)}.png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

sns.set_theme(style="ticks", font="Arial")
plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False, "mathtext.default": "regular"})
corr_data = pd.read_excel(FILE_PATH)[NUM_FEATURES + [TARGET_COL]].rename(columns={k: PRETTY_LABELS[k] for k in NUM_FEATURES + [TARGET_COL]})
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data.corr(method="spearman"), annot=True, fmt=".2f", cmap=sns.diverging_palette(220, 40, s=80, l=55, as_cmap=True), linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 18})
plt.xticks(rotation=20, ha="center", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "spearman_heatmap_ZIF8.png"), dpi=600, bbox_inches="tight")
plt.close()

sns.set_style("ticks")
plt.rcParams.update({"font.family": "Arial", "font.size": 18, "axes.labelsize": 22, "xtick.labelsize": 16, "ytick.labelsize": 16, "mathtext.fontset": "stixsans", "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5})
for x_col in CAT_FEATURES:
    fig, ax = plt.subplots(figsize=(7, 6))
    categories = df_vis[x_col].unique()
    parts = ax.violinplot([df_vis[df_vis[x_col] == cat][TARGET_COL].values for cat in categories], positions=range(len(categories)), widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    colors = sns.color_palette(PALETTE, len(categories))
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_linewidth(1.5)
        pc.set_alpha(0.3)
        pc.get_paths()[0].vertices[:, 1] = np.maximum(0, pc.get_paths()[0].vertices[:, 1])
    for j, cat in enumerate(categories):
        y_data = df_vis[df_vis[x_col] == cat][TARGET_COL].values
        ax.scatter(np.random.normal(j, 0.04, size=len(y_data)), y_data, alpha=0.7, s=30, color=colors[j], edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel(PRETTY_LABELS.get(x_col, x_col))
    ax.set_ylabel(PRETTY_LABELS.get(TARGET_COL, TARGET_COL))
    ax.set_ylim(-df_vis[TARGET_COL].max() * 0.05, df_vis[TARGET_COL].max() * 1.15)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="white", markerfacecolor="gray", markersize=8, linestyle="", label="Samples", alpha=0.6, markeredgecolor="white", markeredgewidth=0.3)], loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    filename = f"violin_plot_{x_col.replace(' ', '_')}"
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f"{filename}.png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close()


# %%
# Model training


# %%
# Part 2. Tree-model tuning and train/test R?
ensure_dir(PLOT_OUTPUT_DIR)
ensure_dir(RF_OUTPUT_DIR)

training_summary = []

X_rf, y_rf, _ = load_xy(filter_positive=True)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=60)
pipe_rf = Pipeline([("pre", build_tree_preprocess(NUM_FEATURES, CAT_FEATURES)), ("reg", RandomForestRegressor(random_state=42))])
param_dist_rf = {
    "reg__n_estimators": np.arange(100, 501, 50),
    "reg__max_depth": np.arange(2, 15, 2),
    "reg__min_samples_split": np.arange(2, 15, 2),
    "reg__min_samples_leaf": np.arange(1, 7, 1),
    "reg__max_features": ["sqrt", "log2"],
    "reg__ccp_alpha": np.round(np.logspace(-4, -2, 10), 4),
}
search_rf = RandomizedSearchCV(pipe_rf, param_dist_rf, n_iter=100, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2", n_jobs=-1, verbose=0, random_state=42)
search_rf.fit(X_train_rf, y_train_rf)
best_rf = search_rf.best_estimator_
print("\nRandomForest best parameters:")
print(search_rf.best_params_)
rf_scores = print_r2(best_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf)
training_summary.append({"Model": "RandomForest", "Best parameters": search_rf.best_params_, **rf_scores})
joblib.dump(best_rf, os.path.join(RF_OUTPUT_DIR, "rf_zif8_model_current.pkl"))

X, y, df_model = load_xy(filter_positive=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
pipe_xgb = Pipeline([("pre", build_tree_preprocess(num_features, cat_features)), ("reg", XGBRegressor(random_state=42, objective="reg:squarederror"))])
param_dist_xgb = {
    "reg__n_estimators": [100, 150, 200, 250, 300, 350, 400],
    "reg__learning_rate": [0.01, 0.03, 0.05, 0.06, 0.07, 0.08],
    "reg__max_depth": [3, 4, 5],
    "reg__subsample": [0.65, 0.7, 0.75, 0.8],
    "reg__colsample_bytree": [0.65, 0.7, 0.75, 0.8],
    "reg__colsample_bylevel": [0.65, 0.7, 0.75, 0.8],
    "reg__reg_lambda": [5, 10, 20, 30, 50],
    "reg__reg_alpha": [0.01, 0.1, 0.5, 1, 5],
    "reg__min_child_weight": [2, 3, 4, 5, 6],
    "reg__gamma": [0.1, 0.3, 0.5, 1, 1.5],
}
search_xgb = RandomizedSearchCV(pipe_xgb, param_dist_xgb, n_iter=100, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2", n_jobs=-1, verbose=0, random_state=42)
search_xgb.fit(X_train, y_train)
best_xgb = search_xgb.best_estimator_
print("\nXGBoost best parameters:")
print(search_xgb.best_params_)
xgb_scores = print_r2(best_xgb, X_train, X_test, y_train, y_test)
training_summary.append({"Model": "XGBoost", "Best parameters": search_xgb.best_params_, **xgb_scores})
joblib.dump(best_xgb, os.path.join(PLOT_OUTPUT_DIR, "xgb_zif8_model_current.pkl"))

preprocessor = ColumnTransformer([("num_std", StandardScaler(), num_features)], remainder="passthrough")
cat_features_idx = list(range(len(num_features), len(num_features) + len(cat_features)))
param_dist_cat = {
    "reg__learning_rate": [round(x, 2) for x in np.linspace(0.01, 0.05, 20)],
    "reg__depth": [4, 6, 8],
    "reg__l2_leaf_reg": [round(x, 1) for x in np.linspace(1.0, 10.0, 10)],
    "reg__bagging_temperature": [round(x, 1) for x in np.linspace(0.0, 2.0, 5)],
}
pipe_cat = Pipeline([("pre", preprocessor), ("reg", CatBoostRegressor(iterations=200, random_seed=42, verbose=False, cat_features=cat_features_idx, thread_count=-1))])
search_cat = RandomizedSearchCV(pipe_cat, param_dist_cat, n_iter=100, scoring="r2", cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1, verbose=0, random_state=42)
search_cat.fit(X_train, y_train)
best_catboost_model = search_cat.best_estimator_
print("\nCatBoost best parameters:")
print(search_cat.best_params_)
cat_scores = print_r2(best_catboost_model, X_train, X_test, y_train, y_test)
training_summary.append({"Model": "CatBoost", "Best parameters": search_cat.best_params_, **cat_scores})
joblib.dump(best_catboost_model, BEST_CATBOOST_SAVE_PATH)
print(f"\nCurrent CatBoost model saved to: {BEST_CATBOOST_SAVE_PATH}")

training_summary_df = pd.DataFrame(training_summary)
training_summary_df.to_excel(os.path.join(PLOT_OUTPUT_DIR, "tree_model_training_r2_summary.xlsx"), index=False)
display(training_summary_df[["Model", "Best parameters", "Train R2", "Test R2"]])


# %%
# SHAP analysis


# %%
# Part 3. CatBoost SHAP analysis and feature-dependence plots
if shap is None:
    raise ImportError("Please install shap first: pip install shap")
ensure_dir(SHAP_OUTPUT_DIR)
ensure_dir(DEPENDENCE_OUTPUT_DIR)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial"

best_model = best_catboost_model
preprocessor, model = best_model.named_steps["pre"], best_model.named_steps["reg"]
all_features = list(X.columns)
X_proc = preprocessor.transform(X)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_proc)
if isinstance(shap_values, list): shap_values = shap_values[0]

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
if isinstance(shap_values_dep, list): shap_values_dep = shap_values_dep[0]
unit_map = {"CZn": " (mol/L)", "Solvent volume": " (mL)", "Temperature": " (°C)", "Reaction time": " (min)"}
for i, feat in enumerate(num_features + cat_features):
    plt.figure(figsize=(6.5, 5))
    if feat in num_features:
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


# %%
# Prediction on new experiments


# %%
# Part 4. Experimental-validation prediction using the current CatBoost model
loaded_model = best_catboost_model
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
predictions = loaded_model.predict(new_data[num_features + cat_features])

print("\n" + "=" * 60)
print(f"{'Recipe':<8} | {'Solvent':<10} | {'Stirring':<10} | {'Predicted Size (nm)':<20}")
print("-" * 60)
for i, size in enumerate(predictions, 1):
    print(f"{i:<8} | {new_data.iloc[i-1]['Solvent type']:<10} | {new_data.iloc[i-1]['Stirring']:<10} | {size:>10.2f} nm")
print("=" * 60)

new_data["Predicted_Size_nm"] = predictions
new_data.to_excel(os.path.join(PLOT_OUTPUT_DIR, "ZIF8_experimental_validation_predictions_current_catboost.xlsx"), index=False)

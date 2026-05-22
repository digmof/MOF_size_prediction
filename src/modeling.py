import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import BEST_CATBOOST_SAVE_PATH, CAT_FEATURES, MODEL_OUTPUT_DIR, NUM_FEATURES, PLOT_OUTPUT_DIR
from data import load_xy
from utils import ensure_dir


def to_numeric_func(X_):
    return pd.DataFrame(X_).apply(pd.to_numeric, errors="coerce")


def to_str_func(X_):
    return pd.DataFrame(X_).astype(str)


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


def train_tree_models():
    ensure_dir(PLOT_OUTPUT_DIR)
    ensure_dir(MODEL_OUTPUT_DIR)
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
    joblib.dump(best_rf, os.path.join(MODEL_OUTPUT_DIR, "rf_zif8_model_current.pkl"))

    X, y, df_model = load_xy(filter_positive=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    pipe_xgb = Pipeline([("pre", build_tree_preprocess(NUM_FEATURES, CAT_FEATURES)), ("reg", XGBRegressor(random_state=42, objective="reg:squarederror"))])
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

    preprocessor = ColumnTransformer([("num_std", StandardScaler(), NUM_FEATURES)], remainder="passthrough")
    cat_features_idx = list(range(len(NUM_FEATURES), len(NUM_FEATURES) + len(CAT_FEATURES)))
    param_dist_cat = {
        "reg__learning_rate": [round(x, 2) for x in np.linspace(0.01, 0.05, 20)],
        "reg__depth": [4, 6, 8],
        "reg__l2_leaf_reg": [round(x, 1) for x in np.linspace(1.0, 10.0, 10)],
        "reg__bagging_temperature": [round(x, 1) for x in np.linspace(0.0, 2.0, 5)],
    }
    pipe_cat = Pipeline([("pre", preprocessor), ("reg", CatBoostRegressor(iterations=200, random_seed=42, verbose=False, thread_count=-1))])
    search_cat = RandomizedSearchCV(pipe_cat, param_dist_cat, n_iter=100, scoring="r2", cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1, verbose=0, random_state=42)
    search_cat.fit(X_train, y_train, reg__cat_features=cat_features_idx)
    best_catboost_model = search_cat.best_estimator_
    print("\nCatBoost best parameters:")
    print(search_cat.best_params_)
    cat_scores = print_r2(best_catboost_model, X_train, X_test, y_train, y_test)
    training_summary.append({"Model": "CatBoost", "Best parameters": search_cat.best_params_, **cat_scores})
    joblib.dump(best_catboost_model, BEST_CATBOOST_SAVE_PATH)
    print(f"\nCurrent CatBoost model saved to: {BEST_CATBOOST_SAVE_PATH}")

    training_summary_df = pd.DataFrame(training_summary)
    training_summary_df.to_excel(os.path.join(PLOT_OUTPUT_DIR, "tree_model_training_r2_summary.xlsx"), index=False)
    return best_catboost_model, X, X_train, training_summary_df

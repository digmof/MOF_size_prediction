import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from config import (
    ACADEMIC_BLUE,
    CAT_FEATURES,
    FILE_PATH,
    NUM_FEATURES,
    PALETTE,
    PLOT_OUTPUT_DIR,
    PRETTY_LABELS,
    SHEET_NAME,
    TARGET_COL,
)
from utils import ensure_dir, safe_name


def style_axes(ax, lw=1.1):
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(lw)


def plot_database_distribution():
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

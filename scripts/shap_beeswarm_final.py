"""
=======================================================================
SHAP BEESWARM PLOTS — ALL 4 CITIES × 3 YEARS + AVERAGED
=======================================================================
SHAP version compatibility:
  Old SHAP (<0.40): shap_values is list of (n_test, n_features) per class
  New SHAP (>=0.40): shap_values is ndarray of (n_test, n_features, n_classes)
  This script handles BOTH automatically.

OUTPUT:
  Individual: SHAP_Beeswarm_Outputs/SHAP_Beeswarm_{City}_{Year}.png
  Averaged:   SHAP_Beeswarm_Outputs/SHAP_Beeswarm_{City}_Averaged.png

CLASS MAPPING:
  0 = Vegetation | 1 = Built-up | 2 = Water | 3 = Others
=======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shap
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ===================================================================
# GLOBAL FONT — ALL BOLD THROUGHOUT
# ===================================================================

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.weight":           "bold",
    "axes.titleweight":      "bold",
    "axes.labelweight":      "bold",
    "axes.titlesize":        15,
    "axes.labelsize":        12,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       11,
    "figure.titlesize":      22,
    "figure.titleweight":    "bold",
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
    "axes.linewidth":        1.2,
})

# ===================================================================
# CONFIGURATION
# ===================================================================

CITIES = ["Hyderabad", "Ahmedabad", "Pune", "Surat"]
YEARS  = [2015, 2020, 2025]

CSV_NAMES = {
    ("Hyderabad",  2015): "RF_SHAP_Data_Hyderabad_2015.csv",
    ("Hyderabad",  2020): "RF_SHAP_Data_Hyderabad_2020.csv",
    ("Hyderabad",  2025): "RF_SHAP_Data_Hyderabad_2025.csv",
    ("Ahmedabad",  2015): "RF_SHAP_Data_Ahmedabad_2015.csv",
    ("Ahmedabad",  2020): "RF_SHAP_Data_Ahmedabad_2020.csv",
    ("Ahmedabad",  2025): "RF_SHAP_Data_Ahmedabad_2025.csv",
    ("Pune",       2015): "RF_SHAP_Data_Pune_2015.csv",
    ("Pune",       2020): "RF_SHAP_Data_Pune_2020.csv",
    ("Pune",       2025): "RF_SHAP_Data_Pune_2025.csv",
    ("Surat",      2015): "RF_SHAP_Data_Surat_2015.csv",
    ("Surat",      2020): "RF_SHAP_Data_Surat_2020.csv",
    ("Surat",      2025): "RF_SHAP_Data_Surat_2025.csv",
}

BANDS = [
    "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
    "NDVI", "NDBI", "NDWI", "SAVI", "UI",
    "TC1", "TC2", "TC3", "BCI",
    "GLCM_contrast", "GLCM_entropy", "EBBI"
]

# ── class label 4 = Others (not 3) ──────────────────────────────────
CLASS_MAP   = {0: "Vegetation", 1: "Built-up", 2: "Water", 3: "Others"}
CLASS_ORDER = [0, 1, 2, 3]

CLASS_COLORS = {
    "Vegetation": "#2E7D32",   # dark green
    "Built-up":   "#B71C1C",   # dark red
    "Water":      "#0D47A1",   # dark blue
    "Others":     "#E65100",   # deep orange
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

SEED    = 42
N_TREES = 500
DPI     = 600


os.makedirs("SHAP_Beeswarm_Outputs", exist_ok=True)


# ===================================================================
# HELPER: make ALL text in axes bold after shap.summary_plot
# ===================================================================

def make_axes_bold(ax):
    """
    shap.summary_plot resets many font weights internally.
    This function re-applies bold to everything after the plot is drawn.
    """
    # Axis tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(11)

    # Axis labels
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(12)

    # Title
    ax.title.set_fontweight("bold")
    ax.title.set_fontsize(15)

    # Colorbar text (if present — shap adds it as a separate axes)
    for child_ax in ax.get_figure().get_axes():
        if child_ax is not ax:
            for label in child_ax.get_yticklabels() + child_ax.get_xticklabels():
                label.set_fontweight("bold")
                label.set_fontsize(10)
            if child_ax.yaxis.label.get_text():
                child_ax.yaxis.label.set_fontweight("bold")
                child_ax.yaxis.label.set_fontsize(10)
            if child_ax.xaxis.label.get_text():
                child_ax.xaxis.label.set_fontweight("bold")


# ===================================================================
# HELPER: normalise shap output to dict {class_label: (n_test, n_feat)}
# ===================================================================

def normalise_shap(shap_raw, rf_classes):
    shap_raw = np.array(shap_raw)

    # New format: (n_test, n_features, n_classes)
    if shap_raw.ndim == 3 and shap_raw.shape[2] == len(rf_classes):
        return {int(cls): shap_raw[:, :, i] for i, cls in enumerate(rf_classes)}

    # Old format: (n_classes, n_test, n_features)
    if shap_raw.ndim == 3 and shap_raw.shape[0] == len(rf_classes):
        return {int(cls): shap_raw[i] for i, cls in enumerate(rf_classes)}

    raise ValueError(
        f"Unexpected shap_values shape: {shap_raw.shape} "
        f"for {len(rf_classes)} classes"
    )


# ===================================================================
# HELPER: train RF and return normalised shap dict + X_test
# ===================================================================

def get_shap_dict(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["class"].isin(CLASS_ORDER)].reset_index(drop=True)

    X = df[BANDS].values
    y = df["class"].values

    X_train, X_test, y_train, _ = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_features=4,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_raw  = explainer.shap_values(X_test)

    shap_dict = normalise_shap(shap_raw, list(rf.classes_))
    X_test_df = pd.DataFrame(X_test, columns=BANDS)

    return shap_dict, X_test_df


# ===================================================================
# HELPER: draw one 4-panel beeswarm figure
# ===================================================================

def draw_beeswarm(shap_dict, X_test_df, suptitle, save_path,
                  xlabel_suffix=""):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(22, 18),
        facecolor="white"
    )

    # Main title — bold, large
    fig.suptitle(
        suptitle,
        fontsize=22,
        fontweight="bold",
        y=1.01,
        color="black"
    )

    for idx, cls_label in enumerate(CLASS_ORDER):

        ax         = axes.flatten()[idx]
        class_name = CLASS_MAP[cls_label]
        color      = CLASS_COLORS[class_name]

        plt.sca(ax)

        if cls_label not in shap_dict:
            ax.text(
                0.5, 0.5,
                f"{class_name}\nnot in data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=13, fontweight="bold"
            )
            ax.set_title(
                class_name,
                fontsize=15, fontweight="bold", color=color
            )
            continue

        shap_mat = shap_dict[cls_label]   # (n_test, n_features)

        assert shap_mat.shape[1] == len(BANDS), (
            f"Shape mismatch: shap {shap_mat.shape}, "
            f"features {len(BANDS)}"
        )

        # Draw beeswarm
        shap.summary_plot(
            shap_mat,
            X_test_df,
            feature_names=BANDS,
            show=False,
            plot_size=None,
            color_bar=True,
            max_display=18,
            plot_type="dot"
        )

        # ── Panel title ─────────────────────────────────────────────
        ax.set_title(
            class_name,
            fontsize=15,
            fontweight="bold",
            color=color,
            pad=10
        )

        # ── X-axis label ────────────────────────────────────────────
        ax.set_xlabel(
            f"SHAP value  (impact on classification probability"
            f"{xlabel_suffix})",
            fontsize=12,
            fontweight="bold",
            labelpad=8
        )

        # ── Y-axis label ────────────────────────────────────────────
        ax.set_ylabel(
            "Feature",
            fontsize=12,
            fontweight="bold",
            labelpad=8
        )

        # ── Panel label (a)(b)(c)(d) ────────────────────────────────
        ax.text(
            -0.10, 1.03,
            PANEL_LABELS[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black"
        )

        # ── Spines ──────────────────────────────────────────────────
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # ── Re-apply bold to everything shap reset ──────────────────
        make_axes_bold(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4, w_pad=3)
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close("all")
    print(f"    Saved → {os.path.basename(save_path)}")


# ===================================================================
# MAIN LOOP
# ===================================================================

print("=" * 70)
print("SHAP BEESWARM GENERATOR — FINAL VERSION")
print("=" * 70)

for city in CITIES:

    print(f"\n{'─'*70}")
    print(f"  CITY: {city}")
    print(f"{'─'*70}")

    year_shap_dicts = {}

    # ── Individual year plots ───────────────────────────────────────
    for year in YEARS:

        csv_file = CSV_NAMES.get((city, year), "")

        if not os.path.exists(csv_file):
            print(f"  ⚠  Not found: {csv_file}")
            continue

        print(f"  Processing {year}...", end="", flush=True)

        try:
            shap_dict, X_test_df = get_shap_dict(csv_file)
            year_shap_dicts[year] = (shap_dict, X_test_df)

            save_path = os.path.join(
                "SHAP_Beeswarm_Outputs",
                f"SHAP_Beeswarm_{city}_{year}.png"
            )

            draw_beeswarm(
                shap_dict,
                X_test_df,
                suptitle=f"{city} {year}  —  SHAP Beeswarm "
                         f"(RF Classification)",
                save_path=save_path,
                xlabel_suffix=""
            )

        except Exception as e:
            print(f"\n    ✗ Error: {e}")

    # ── Averaged plot ───────────────────────────────────────────────
    if len(year_shap_dicts) == 0:
        print(f"  ✗ No years succeeded — skipping averaged plot")
        continue

    print(f"  Building averaged plot ({len(year_shap_dicts)} years)...")

    # Pool raw SHAP and X per class across all years
    pooled_shap = {cls: [] for cls in CLASS_ORDER}
    pooled_X    = {cls: [] for cls in CLASS_ORDER}

    for year, (sd, xdf) in year_shap_dicts.items():
        for cls in CLASS_ORDER:
            if cls in sd:
                pooled_shap[cls].append(sd[cls])
                pooled_X[cls].append(xdf.values)

    avg_shap_dict  = {}
    avg_X_dict     = {}

    for cls in CLASS_ORDER:
        if pooled_shap[cls]:
            avg_shap_dict[cls] = np.vstack(pooled_shap[cls])
            avg_X_dict[cls]    = pd.DataFrame(
                np.vstack(pooled_X[cls]),
                columns=BANDS
            )

    # ── Draw averaged — pass per-class X to each panel ─────────────
    fig, axes = plt.subplots(
        2, 2,
        figsize=(22, 18),
        facecolor="white"
    )

    fig.suptitle(
        f"{city}  —  Class-specific spectral feature contributions to RF Classification (2015–2025)",
        fontsize=22,
        fontweight="bold",
        y=1.01,
        color="black"
    )

    for idx, cls_label in enumerate(CLASS_ORDER):

        ax         = axes.flatten()[idx]
        class_name = CLASS_MAP[cls_label]
        color      = CLASS_COLORS[class_name]

        plt.sca(ax)

        if cls_label not in avg_shap_dict:
            ax.text(
                0.5, 0.5,
                f"{class_name}\nnot available",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=13, fontweight="bold"
            )
            ax.set_title(
                class_name,
                fontsize=15, fontweight="bold", color=color
            )
            continue

        shap_mat = avg_shap_dict[cls_label]
        X_mat    = avg_X_dict[cls_label]

        shap.summary_plot(
            shap_mat,
            X_mat,
            feature_names=BANDS,
            show=False,
            plot_size=None,
            color_bar=True,
            max_display=18,
            plot_type="dot"
        )

        # ── Panel title ─────────────────────────────────────────────
        ax.set_title(
            class_name,
            fontsize=15,
            fontweight="bold",
            color=color,
            pad=10
        )

        # ── X-axis label ────────────────────────────────────────────
        ax.set_xlabel(
            "SHAP value  (mean absolute contribution to classification probability, 2015–2025)",
            fontsize=12,
            fontweight="bold",
            labelpad=8
        )

        # ── Y-axis label ────────────────────────────────────────────
        ax.set_ylabel(
            "Feature",
            fontsize=12,
            fontweight="bold",
            labelpad=8
        )

        # ── Panel label ─────────────────────────────────────────────
        ax.text(
            -0.10, 1.03,
            PANEL_LABELS[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black"
        )

        # ── Spines ──────────────────────────────────────────────────
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # ── Re-apply bold ────────────────────────────────────────────
        make_axes_bold(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4, w_pad=3)

    avg_save = os.path.join(
        "SHAP_Beeswarm_Outputs",
        f"SHAP_Beeswarm_{city}_Averaged.png"
    )
    plt.savefig(
        avg_save,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close("all")
    print(f"    Saved → SHAP_Beeswarm_{city}_Averaged.png")


print("\n" + "=" * 70)
print("ALL DONE")
print("Output folder: SHAP_Beeswarm_Outputs/")
print("  12 individual year plots  +  4 averaged city plots")
print("=" * 70)

"""
Shared visualization script for genomic and clinical pipelines.
Generates model comparison, per-label F1 heatmap, fairness audit,
and class distribution figures.

To use: python scripts/visualize.py -f [genomic/clinical/combined]
"""
import argparse
import os
import sys
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from model_utils import load_train_test
# config
LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']
target_cols = ['EVER_MET_SITE_ADRENAL', 'EVER_MET_SITE_BONE', 'EVER_MET_SITE_CNS',
               'EVER_MET_SITE_LIVER_BILIARY_TRACT', 'EVER_MET_SITE_LN', 
               'EVER_MET_SITE_LUNG', 'EVER_MET_SITE_PLEURA']
label_map = {
    'test_f1_adrenal': 'Adrenal',
    'test_f1_bone': 'Bone',
    'test_f1_cns': 'CNS',
    'test_f1_liver': 'Liver',
    'test_f1_ln': 'LN',
    'test_f1_lung': 'Lung',
    'test_f1_pleura': 'Pleura'
}

# clean display names for models
MODEL_DISPLAY_NAMES = {
    'logistic': 'Baseline LR',
    'logistic_ridge': 'Ridge (L2)',
    'logistic_lasso': 'Lasso (L1)',
    'LDA': 'LDA',
    'random_forest': 'Random Forest',
    'SVM': 'SVM',
    'MLP': 'MLP'
}

# color palette
COLORS = {
    'test': '#2C7BB6',
    'train': '#ABD9E9',
    'gap': '#D7191C',
    'overall': '#2C7BB6',
    'subgroup': '#74ADD1',
    'positive': '#1A9641',
    'negative': '#D7191C',
}

FIGURE_DPI = 150
OUTPUT_DIR = 'results/figures'


# helper function
def load_eval(feature_set: str) -> pd.DataFrame:
    """Load evaluation CSV and add MLP results if genomic."""
    path = f'results/evaluation/multilabel/evaluation_{feature_set}.csv'
    df = pd.read_csv(path, index_col=0)

    # need to merge MLP results if genomic pipeline since in separate script
    mlp_path = f'results/evaluation/multilabel/evaluation_{feature_set}_mlp.csv'
    if os.path.exists(mlp_path):
        mlp = pd.read_csv(mlp_path, index_col=0)
        df = pd.concat([df, mlp], axis=1)

    # rename columns to clean display names
    df.columns = [MODEL_DISPLAY_NAMES.get(c, c) for c in df.columns]
    return df


def load_models(feature_set: str) -> dict:
    '''loads a fitted model'''
    # load from feature-set specific subfolder
    path = Path(f'models/fitted_models/{feature_set}/binary')
    models = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix == '.pkl' and not item.stem.startswith('gridsearch'):
            model = joblib.load(str(item))
            model_name = item.stem.replace(f'_{feature_set}', '')
            models[model_name] = model
    return models

def save_figure(fig, name: str, feature_set: str):
    """Save figure to results/figures/."""
    if feature_set not in ['clinical', 'genomic', 'combined']:
        dir = f'{OUTPUT_DIR}/other'

    else: dir = f'{OUTPUT_DIR}/{feature_set}/'
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f'{feature_set}_{name}.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


# Figure 1: Model comparison bar chart
def plot_model_comparison(df: pd.DataFrame, feature_set: str, ax=None):
    """
    Side by side bar chart of test F1 and train F1 per model.
    Sorted by test F1 descending. Overfitting gap visible as difference between bars.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    models = df.columns.tolist()
    test_f1 = df.loc['test_f1'].values
    train_f1 = df.loc['train_f1'].values

    x = np.arange(len(models))
    width = 0.35

    # bars
    bars_test = ax.bar(x - width/2, test_f1, width,
                       label='Test F1', color=COLORS['test'], alpha=0.9)
    bars_train = ax.bar(x + width/2, train_f1, width,
                        label='Train F1', color=COLORS['train'], alpha=0.9)

    # ---- value labels ----
    for bar, val in zip(bars_test, test_f1):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() - 0.03,
            f'{val:.3f}',
            ha='center',
            va='top',
            fontsize=8,
            fontweight='bold',
            color='black',
            rotation=90
        )
    for bar, val in zip(bars_train, train_f1):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() - 0.03,
            f'{val:.3f}',
            ha='center',
            va='top',
            fontsize=8,
            fontweight='bold',
            color='black',
            rotation=90
        )
    ax.set_title(f'{feature_set.capitalize()} Features', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)

    ax.axhline(y=0, linewidth=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

# Figure 1: Per-label F1 heatmap
def plot_per_label_heatmap(feature_set: str, ax=None):
    """
    Heatmap of per-label F1 scores across all models.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure  # get parent figure

    # --- Load data ---
    genomic_per_label = load_eval('genomic')
    genomic_per_label = genomic_per_label[genomic_per_label.index.str.startswith('test_f1_')]
    genomic_per_label.index = genomic_per_label.index.map(label_map)

    clinical_per_label = load_eval('clinical')
    clinical_per_label = clinical_per_label[clinical_per_label.index.str.startswith('test_f1_')]
    clinical_per_label.index = clinical_per_label.index.map(label_map)

    combined_per_label = load_eval('combined')
    combined_per_label = combined_per_label[combined_per_label.index.str.startswith('test_f1_')]
    combined_per_label.index = combined_per_label.index.map(label_map)

    combined_interact_per_label = load_eval('combined_interact')
    combined_interact_per_label = combined_interact_per_label[combined_interact_per_label.index.str.startswith('test_f1_')]
    combined_interact_per_label.index = combined_interact_per_label.index.map(label_map)

    per_label_data = {
    'clinical': clinical_per_label,
    'genomic': genomic_per_label,
    'combined': combined_per_label,
    'combined_interact': combined_interact_per_label}

    data = per_label_data[feature_set]
    df = pd.DataFrame(data, index=LABEL_NAMES)

    # --- Colormap ---
    cmap = LinearSegmentedColormap.from_list(
        'f1_cmap', ['#FFFFFF', '#ABD9E9', '#2C7BB6', '#08306B']
    )

    # --- Plot ---
    im = ax.imshow(df.values, aspect='auto', cmap=cmap, vmin=0, vmax=0.7)

    # annotate cells
    for i in range(len(LABEL_NAMES)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            color = 'white' if val > 0.45 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=15, ha='right', fontsize=10)
    # ax.set_yticks(range(len(LABEL_NAMES)))
    # ax.set_yticklabels(LABEL_NAMES, fontsize=10)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_title(f'{feature_set.capitalize()} Features', fontsize=12)

    # --- Colorbar (important tweak for panels) ---
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax

# Figure 2: Class distribution
def plot_class_distribution():
    """
    Bar chart of label distribution in y.csv.
    Shared across both pipelines — same dataset.
    Only generated once.
    """
    y = pd.read_csv('data/processed/y_multilabel.csv')
    y = y.drop(columns=[c for c in ['sampleId', 'patientId'] if c in y.columns])

    counts = y.sum().values
    labels = LABEL_NAMES
    total = len(y)

    # also add non-metastatic count (samples with all zeros)
    non_met = (y.sum(axis=1) == 0).sum()
    counts = np.append(counts, non_met)
    labels = labels + ['Non-Metastatic']

    colors = ['#2C7BB6'] * 7 + ['#74ADD1']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.9, edgecolor='white', linewidth=0.8)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xlabel('Metastatic Site / Class', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Label Distribution in Dataset\n'
                 '(patients may appear in multiple metastatic site columns)',
                 fontsize=13)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, 'class_distribution', 'shared')


# Figure 3: Fairness audit
def plot_fairness_audit_comparison():
    """
    Combined fairness visualization showing:
    1. Overall F1 across all three pipelines per subgroup
    2. Gap from overall as a secondary view
    """
    # results from fairness_audit_all.py
    data = {
        'Clinical':  {'overall': 0.348, 'Race: White': 0.360, 'Sex: Male': 0.397, 'Sex: Female': 0.325},
        'Genomic':   {'overall': 0.240, 'Race: White': 0.235, 'Sex: Male': 0.274, 'Sex: Female': 0.226},
        'Combined':  {'overall': 0.339, 'Race: White': 0.327, 'Sex: Male': 0.357, 'Sex: Female': 0.341},
    }

    subgroups = ['Race: White', 'Sex: Male', 'Sex: Female']
    pipelines = list(data.keys())
    x = np.arange(len(subgroups))
    width = 0.25
    pipeline_colors = ['#2C7BB6', '#74ADD1', '#ABD9E9']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: absolute F1 per subgroup per pipeline ───────────────────────────
    for i, (pipeline, color) in enumerate(zip(pipelines, pipeline_colors)):
        vals = [data[pipeline][s] for s in subgroups]
        bars = ax1.bar(x + i*width, vals, width, label=pipeline,
                       color=color, alpha=0.9, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # overall F1 reference lines per pipeline
    for pipeline, color, ls in zip(pipelines, pipeline_colors, ['-', '--', ':']):
        ax1.axhline(y=data[pipeline]['overall'], color=color, linewidth=1.5,
                    linestyle=ls, alpha=0.7, label=f'{pipeline} overall ({data[pipeline]["overall"]:.3f})')

    ax1.set_xlabel('Demographic Subgroup', fontsize=11)
    ax1.set_ylabel('Macro F1 Score', fontsize=11)
    ax1.set_title('Subgroup F1 by Feature Set', fontsize=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(subgroups, fontsize=10)
    ax1.set_ylim(0, 0.55)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right: gap from overall per subgroup per pipeline ─────────────────────
    for i, (pipeline, color) in enumerate(zip(pipelines, pipeline_colors)):
        gaps = [data[pipeline][s] - data[pipeline]['overall'] for s in subgroups]
        bars = ax2.bar(x + i*width, gaps, width, label=pipeline,
                       color=[COLORS['positive'] if g >= 0 else COLORS['negative'] for g in gaps],
                       alpha=0.85, edgecolor='white')
        for bar, gap in zip(bars, gaps):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.003 if gap >= 0 else -0.008),
                     f'{gap:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.axhline(y=0, color='black', linewidth=1.2, linestyle='-')
    ax2.set_xlabel('Demographic Subgroup', fontsize=11)
    ax2.set_ylabel('Gap from Overall F1', fontsize=11)
    ax2.set_title('Fairness Gap by Feature Set\n(positive = above overall, negative = below)', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(subgroups, fontsize=10)
    ax2.legend(pipelines, fontsize=9, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # summary annotation
    fig.suptitle('Fairness Audit — Model Performance Across Demographic Subgroups\n'
                 'Ridge (L2) evaluated on Clinical, Genomic, and Combined Feature Sets',
                 fontsize=13, y=1.02)

    fig.tight_layout()
    save_figure(fig, 'fairness_audit_comparison', 'all')


# Figure 4: Gene mutation frequency (genomic only)
def plot_gene_frequency():
    """
    Horizontal bar chart of mutation frequency per gene across 455 samples.
    Shows the long tail of rare mutations and dominant LUAD drivers.
    Genomic pipeline only.
    """
    freq_data = {
        'TP53': 40.7, 'KRAS': 37.6, 'EGFR': 33.6, 'STK11': 19.6,
        'RBM10': 15.2, 'KEAP1': 13.6, 'NF1': 7.7, 'ATM': 7.7,
        'NTRK3': 7.3, 'SMARCA4': 6.8, 'SETD2': 6.6, 'BRAF': 6.2,
        'PIK3CA': 4.8, 'ARID1A': 4.6, 'MTOR': 4.6, 'RB1': 4.4,
        'APC': 4.4, 'CDKN2A': 4.2, 'SMAD4': 4.2, 'ERBB2': 4.2,
        'TERT': 4.0, 'NOTCH2': 3.7, 'ALK': 3.5, 'U2AF1': 3.5,
        'CTNNB1': 3.5, 'MET': 3.5, 'BRCA2': 3.3, 'NTRK2': 2.6,
        'ROS1': 2.4, 'NOTCH1': 2.2, 'BRCA1': 2.2, 'TSC2': 2.0,
        'RET': 2.0, 'NTRK1': 1.8, 'PTEN': 1.8, 'TSC1': 1.5,
        'FGFR1': 1.3, 'MAP2K1': 0.9, 'FGFR2': 0.9, 'CDK6': 0.7,
        'MYCN': 0.9
    }

    genes = list(freq_data.keys())
    freqs = list(freq_data.values())

    # color by frequency tier
    colors = ['#08306B' if f >= 10 else '#2C7BB6' if f >= 5 else '#74ADD1'
              for f in freqs]

    fig, ax = plt.subplots(figsize=(8, 12))
    bars = ax.barh(genes[::-1], freqs[::-1], color=colors[::-1],
                   alpha=0.9, edgecolor='white', linewidth=0.5)

    ax.axvline(x=5, color='red', linewidth=1.2, linestyle='--',
               alpha=0.7, label='5% threshold')
    ax.set_xlabel('Mutation Frequency (% of samples)', fontsize=12)
    ax.set_title('Gene Panel Mutation Frequency\n(455 LUAD samples, 38 genes)',
                 fontsize=13)

    # legend for tiers
    high = mpatches.Patch(color='#08306B', label='≥10% (high frequency)')
    mid = mpatches.Patch(color='#2C7BB6', label='5–10% (moderate)')
    low = mpatches.Patch(color='#74ADD1', label='<5% (low, retained for LUAD relevance)')
    ax.legend(handles=[high, mid, low], fontsize=9, loc='lower right')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, 'gene_frequency', 'genomic')

def plot_cooccurence():
    df = pd.read_csv('data/processed/y_multilabel.csv')
    fig, ax = plt.subplots(figsize=(8, 7))
    # compute co-occurrence matrix
    cooccurrence = df[target_cols].T.dot(df[target_cols])
    cooccurrence_norm = cooccurrence.div(cooccurrence.values.diagonal(), axis=0)
    # clean up labels
    labels = [col.replace('EVER_MET_SITE_', '').replace('_', ' ').title() 
            for col in target_cols]

    sns.heatmap(cooccurrence_norm, 
                annot=True, 
                fmt='.2f',
                xticklabels=labels,
                yticklabels=labels,
                cmap='Blues',
                ax=ax)

    ax.set_title('Metastatic Site Co-occurrence Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    ax.set_xlabel('')
    ax.set_ylabel('')
    save_figure(fig, 'heatmap', 'cooccurence')

def plot_pca(feature_set):
    X_train, X_test, y_train, y_test = load_train_test(
        'data/processed/X_combined.csv',
        'data/processed/y_binary.csv',
        scale=True
    )
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Load actual fitted models
    ridge_path = 'models/fitted_models/combined/binary/logistic_ridge_combined.pkl'
    rf_path = 'models/fitted_models/combined/binary/random_forest_combined.pkl'
    try:
        ridge = joblib.load(ridge_path)
        rf = joblib.load(rf_path)
    except FileNotFoundError:
        print("File Not Found")
        sys.exit(1)

    # PCA to 2D, fit on training data only
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Refit loaded models on 2D PCA features
    ridge_2d = ridge.__class__(**ridge.get_params())
    rf_2d = rf.__class__(**rf.get_params())
    ridge_2d.fit(X_train_2d, y_train)
    rf_2d.fit(X_train_2d, y_train)

    # Mesh grid
    x_min, x_max = X_train_2d[:, 0].min() - 2, X_train_2d[:, 0].max() + 2
    y_min, y_max = X_train_2d[:, 1].min() - 2, X_train_2d[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                        np.linspace(y_min, y_max, 300))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_2d, title in zip(axes,
                                    [ridge_2d, rf_2d],
                                    ['Ridge (L2) — Linear Boundary',
                                    'Random Forest — Nonlinear Boundary']):
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                            c=y_test, cmap='RdBu', edgecolors='k',
                            linewidth=0.5, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # discrete legend instead of colorbar
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                            markersize=10, label='Never Metastasized'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                            markersize=10, label='Ever Metastasized')]
    axes[1].legend(handles=legend_elements, loc='upper right')

    plt.suptitle('Decision Boundaries in PCA Space — Binary Classification',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'boundary_pca', 'decision')

def plot_multilabel_panel():
    df_clinical = load_eval('clinical')
    df_genomic = load_eval('genomic')
    df_combined = load_eval('combined')
    df_combined_interact = load_eval('combined_interact')

    fig, axes = plt.subplots(2, 4, figsize=(20,10))

    # ---- Row 1: Model comparison ----
    plot_model_comparison(df_clinical, 'clinical', ax=axes[0, 0])
    plot_model_comparison(df_genomic, 'genomic', ax=axes[0, 1])
    plot_model_comparison(df_combined, 'combined', ax=axes[0, 2])
    plot_model_comparison(df_combined_interact, 'combined_interact', ax=axes[0, 3])
    axes[0][0].set_ylabel('Macro F1 Score', fontsize=11) 

    # ---- Row 2: Heatmaps ----
    plot_per_label_heatmap('clinical', ax=axes[1, 0])
    plot_per_label_heatmap('genomic', ax=axes[1, 1])
    plot_per_label_heatmap('combined', ax=axes[1, 2])
    plot_per_label_heatmap('combined_interact', ax=axes[1, 3])
    axes[1,0].set_yticks(range(len(LABEL_NAMES)))
    axes[1,0].set_yticklabels(LABEL_NAMES, fontsize=10)
    # ---- Subplot labels ----
    titles = [
        "Clinical", "Genomic", "Combined", "Combined w/Interaction",
        "", "", "", ""
    ]

    for i, (ax, title) in enumerate(zip(axes.flatten(), titles)):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(
            -0.12, 1.05,
            f"({chr(65 + i)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top',
            ha='right'
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for title + legend
    save_figure(fig, "panel", 'multilabel')

def plot_ROC_panel(feature_sets: list, task: str, model_dicts: dict):

    fig, axes = plt.subplots(
        1, len(feature_sets),
        figsize=(6 * len(feature_sets), 6),
        sharex=True, sharey=True
    )

    if len(feature_sets) == 1:
        axes = [axes]

    for i, (ax, feature_set) in enumerate(zip(axes, feature_sets)):

        x_path = f'data/processed/X_{feature_set}.csv'
        y_path = f'data/processed/y_binary.csv'
        _, X_test, _, y_test = load_train_test(x_path, y_path, scale=True)
        if task != 'multilabel':
            y_test = y_test.ravel()
        models = model_dicts[feature_set]
        for model_name, model in models.items():
            try:
                y_scores = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_scores = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.2f})")
        # diagonal baseline
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

    titles = ['Clinical', 'Genomic', 'Combined', 'Combined w/ Interaction']
    for i, (title, ax) in enumerate(zip(titles, axes)):
        # title per subplot
        ax.set_title(title, fontsize=11, fontweight='bold')
        # axis limits per subplot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # panel label (A, B, C, D)
        ax.text(
            -0.12, 1.05,
            f"({chr(65 + i)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top',
            ha='right'
        )
        ax.grid(alpha=0.3, linestyle='--')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(
            loc='lower right',
            fontsize=7,
            frameon=False,
            handlelength=1.5,
            labelspacing=0.4
        )

    # axis labels only on edges (cleaner panel style)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    for ax in axes:
        ax.set_xlabel("False Positive Rate", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, 'panel_binary', 'roc')

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_set', '-f', type=str,
                        choices=['genomic', 'clinical', 'combined', 'combined_interact'],
                        required=True)
    
    args = parser.parse_args()
    feature_set = args.feature_set

    print(f'\nGenerating visualizations for: {feature_set}')
    print('=' * 50)

    # load evaluation results
    df = load_eval(feature_set)

    ### plot all figures ###
    # supplementary figures
    plot_cooccurence()
    plot_pca(feature_set)

    #  plot roc curves
    model_dicts = {
    'clinical': load_models('clinical'),
    'genomic': load_models('genomic'),
    'combined': load_models('combined'),
    'combined_interact': load_models('combined_interact')
    }

    plot_ROC_panel(['clinical', 'genomic', 'combined', 'combined_interact'], 'binary', model_dicts)

    # Figure 1 — model comparison and per label heat map
    plot_multilabel_panel()

    # Figure 2 — class distribution (shared, only generate once)
    if feature_set == 'genomic':
        plot_class_distribution()

    # Figure 3 — fairness audit
    if feature_set == 'genomic':  # only generate once
        plot_fairness_audit_comparison()

    # Figure 4 — gene frequency (genomic only)
    if feature_set == 'genomic':
        plot_gene_frequency()

    print(f'\nAll figures saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()

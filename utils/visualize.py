"""
Shared visualization script for genomic and clinical pipelines.
Generates model comparison, per-label F1 heatmap, fairness audit,
and class distribution figures.

To use: python scripts/visualize.py -f [genomic/clinical]
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# config
LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']

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
    path = f'results/evaluation_{feature_set}.csv'
    df = pd.read_csv(path, index_col=0)

    # need to merge MLP results if genomic pipeline since in separate script
    if feature_set == 'genomic':
        mlp_path = 'results/evaluation_genomic_mlp.csv'
        if os.path.exists(mlp_path):
            mlp = pd.read_csv(mlp_path, index_col=0)
            df = pd.concat([df, mlp], axis=1)

    # rename columns to clean display names
    df.columns = [MODEL_DISPLAY_NAMES.get(c, c) for c in df.columns]
    return df


def save_figure(fig, name: str, feature_set: str):
    """Save figure to results/figures/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f'{OUTPUT_DIR}/{feature_set}_{name}.png'
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


# Figure 1: Model comparison bar chart
def plot_model_comparison(df: pd.DataFrame, feature_set: str):
    """
    Side by side bar chart of test F1 and train F1 per model.
    Sorted by test F1 descending. Overfitting gap visible as difference between bars.
    """
    # sort by test F1 descending before extracting arrays
    df = df[df.loc['test_f1'].sort_values(ascending=False).index]

    models = df.columns.tolist()
    test_f1 = df.loc['test_f1'].values

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_test = ax.bar(x - width/2, test_f1, width,
                       label='Test F1', color=COLORS['test'], alpha=0.9)

    for bar, val in zip(bars_test, test_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title(f'Model Performance Comparison — {feature_set.capitalize()} Features\n'
                 f'(bars show test F1 / train F1; gap indicates overfitting)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, 'model_comparison', feature_set)


# Figure 2: Per-label F1 heatmap
# NEED TO UPDATE ONCE WE GET FULL RESULTS
def plot_per_label_heatmap(feature_set: str):
    """
    Heatmap of per-label F1 scores across all models.
    Built from the known results dict for now.
    """
    # per-label F1 data from final runs
    # genomic pipeline results
    genomic_per_label = {
        'Baseline LR': [0.000, 0.348, 0.222, 0.211, 0.483, 0.432, 0.222],
        'Ridge (L2)':  [0.250, 0.429, 0.250, 0.160, 0.600, 0.435, 0.250],
        'Lasso (L1)':  [0.111, 0.400, 0.267, 0.182, 0.552, 0.421, 0.267],
        'LDA':         [0.133, 0.250, 0.133, 0.100, 0.250, 0.300, 0.133],
        'Random Forest':[0.050, 0.337, 0.288, 0.184, 0.445, 0.354, 0.288],
        'SVM':         [0.200, 0.350, 0.240, 0.150, 0.480, 0.400, 0.240],
        'MLP':         [0.111, 0.429, 0.231, 0.000, 0.556, 0.424, 0.231],
    }

    # clinical pipeline results [NEED TO FILL IN]
    clinical_per_label = {
        'Baseline LR':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Ridge (L2)':   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Lasso (L1)':   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'LDA':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Random Forest':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    data = genomic_per_label if feature_set == 'genomic' else clinical_per_label
    df = pd.DataFrame(data, index=LABEL_NAMES)

    # custom colormap: white at 0, deep blue at 1
    cmap = LinearSegmentedColormap.from_list(
        'f1_cmap', ['#FFFFFF', '#ABD9E9', '#2C7BB6', '#08306B']
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(df.values, aspect='auto', cmap=cmap, vmin=0, vmax=0.7)

    # annotate each cell
    for i in range(len(LABEL_NAMES)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            color = 'white' if val > 0.45 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=15, ha='right', fontsize=10)
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_yticklabels(LABEL_NAMES, fontsize=10)
    ax.set_title(f'Per-Label F1 Score Heatmap — {feature_set.capitalize()} Features',
                 fontsize=13, pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'per_label_heatmap', feature_set)


# Figure 3: Class distribution
def plot_class_distribution():
    """
    Bar chart of label distribution in y.csv.
    Shared across both pipelines — same dataset.
    Only generated once.
    """
    y = pd.read_csv('data/processed/y.csv')
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


# Figure 4: Fairness audit
def plot_fairness_audit(feature_set: str):
    """
    Bar chart of macro F1 by demographic subgroup.
    Overall performance shown as reference line.
    Only meaningful subgroups (n >= MIN_SAMPLES) included.
    """
    # results from fairness_audit.py output
    # genomic pipeline
    genomic_fairness = {
        'overall': 0.339,
        'subgroups': {
            'Race: White\n(n=76)': 0.327,
            'Sex: Male\n(n=24)': 0.357,
            'Sex: Female\n(n=67)': 0.341,
        }
    }

    # clinical pipeline - NEED TO FILL IN
    clinical_fairness = {
        'overall': 0.0,
        'subgroups': {}
    }

    data = genomic_fairness if feature_set == 'genomic' else clinical_fairness

    if not data['subgroups']:
        print(f'No fairness data available for {feature_set}, skipping.')
        return

    overall = data['overall']
    subgroups = data['subgroups']
    names = list(subgroups.keys())
    values = list(subgroups.values())
    gaps = [v - overall for v in values]

    # color bars by direction of gap
    bar_colors = [COLORS['positive'] if g >= 0 else COLORS['negative'] for g in gaps]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=bar_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8)

    # overall reference line
    ax.axhline(y=overall, color=COLORS['overall'], linewidth=2,
               linestyle='--', label=f'Overall F1: {overall:.3f}')

    # annotate bars
    for bar, val, gap in zip(bars, values, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}\n({gap:+.3f})', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Demographic Subgroup', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title(f'Fairness Audit — Model Performance by Subgroup\n'
                 f'{feature_set.capitalize()} Features (Ridge, best model)',
                 fontsize=13)
    ax.set_ylim(0, min(1.0, max(values) * 1.25))
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend for bar colors
    pos_patch = mpatches.Patch(color=COLORS['positive'], label='Above overall')
    neg_patch = mpatches.Patch(color=COLORS['negative'], label='Below overall')
    ax.legend(handles=[pos_patch, neg_patch] +
              [mpatches.Patch(color=COLORS['overall'],
               label=f'Overall F1: {overall:.3f}')],
              fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'fairness_audit', feature_set)


# Figure 5: Gene mutation frequency (genomic only)
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


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_set', '-f', type=str,
                        choices=['genomic', 'clinical', 'combined'],
                        required=True)
    args = parser.parse_args()
    feature_set = args.feature_set

    print(f'\nGenerating visualizations for: {feature_set}')
    print('=' * 50)

    # load evaluation results
    df = load_eval(feature_set)

    # Figure 1 — model comparison
    plot_model_comparison(df, feature_set)

    # Figure 2 — per-label heatmap
    plot_per_label_heatmap(feature_set)

    # Figure 3 — class distribution (shared, only generate once)
    if feature_set == 'genomic':
        plot_class_distribution()

    # Figure 4 — fairness audit
    plot_fairness_audit(feature_set)

    # Figure 5 — gene frequency (genomic only)
    if feature_set == 'genomic':
        plot_gene_frequency()

    print(f'\nAll figures saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
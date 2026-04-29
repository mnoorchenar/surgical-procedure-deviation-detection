"""figures.py — Generate all paper figures.

All figures are saved as PDF to results/.
Font size 32 pt throughout. No figure titles. No error bars.
Run from any directory; ROOT is resolved relative to this file.

Usage:
    python code/figures.py
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

SEED = 42
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'outputs', 'result_task3.db')
FIG_DIR = os.path.join(ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size':        32,
    'axes.titlesize':   32,
    'axes.labelsize':   32,
    'xtick.labelsize':  28,
    'ytick.labelsize':  28,
    'legend.fontsize':  24,
    'lines.linewidth':   3.0,
})

# ── Colour palette (one per encoding) ────────────────────────────────────────
ENC_COLOURS = {
    'struct_only':      '#555555',
    'clinicalbert':     '#2255AA',
    'sentencebert':     '#CC4400',
    'rag_clinicalbert': '#228833',
    'rag_sentencebert': '#AA44BB',
}

# Distinct line styles — readable in both colour and greyscale print
ENC_LINESTYLES = {
    'struct_only':      '-',
    'clinicalbert':     '--',
    'sentencebert':     '-.',
    'rag_clinicalbert': ':',
    'rag_sentencebert': (0, (5, 1)),   # densely dashed
}

ENC_LABELS = {
    'struct_only':      'Structured only',
    'clinicalbert':     'ClinicalBERT',
    'sentencebert':     'SentenceBERT',
    'rag_clinicalbert': 'RAG + ClinicalBERT',
    'rag_sentencebert': 'RAG + SentenceBERT',
}

ENC_ORDER = [
    'struct_only', 'clinicalbert', 'sentencebert',
    'rag_clinicalbert', 'rag_sentencebert',
]

MODEL_LABELS = {
    'xgboost':      'XGBoost',
    'mlp':          'MLP',
    'randomforest': 'Random Forest',
    'lightgbm':     'LightGBM',
    'ridge':        'Ridge LR',
}

MODEL_ORDER = ['xgboost', 'mlp', 'randomforest', 'lightgbm', 'ridge']

# Best encoding per model (by CV AUC-ROC)
MODEL_BEST_ENC = {
    'xgboost':      'sentencebert',
    'mlp':          'sentencebert',
    'randomforest': 'sentencebert',
    'lightgbm':     'rag_sentencebert',
    'ridge':        'rag_sentencebert',
}

# Class prevalence for PR baseline (6.77% positive rate)
PR_BASELINE = 0.0677

# Common interpolation grids
FPR_GRID = np.linspace(0, 1, 300)
REC_GRID = np.linspace(0, 1, 300)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_metrics():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM metrics', conn)
    conn.close()
    return df


def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM predictions', conn)
    conn.close()
    return df


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {name}')


def _mean_roc(fold_preds):
    """Return (fpr_grid, mean_tpr) averaged over all folds in fold_preds."""
    tprs = []
    for _, grp in fold_preds.groupby('fold'):
        fpr, tpr, _ = roc_curve(grp['y_true'], grp['y_prob'])
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
    if not tprs:
        return FPR_GRID, np.zeros_like(FPR_GRID)
    return FPR_GRID, np.mean(tprs, axis=0)


def _mean_pr(fold_preds):
    """Return (rec_grid, mean_prec) averaged over all folds in fold_preds."""
    precs = []
    for _, grp in fold_preds.groupby('fold'):
        prec, rec, _ = precision_recall_curve(grp['y_true'], grp['y_prob'])
        # rec from precision_recall_curve is decreasing; reverse for interp
        prec_rev = prec[::-1]
        rec_rev  = rec[::-1]
        # Ensure strictly increasing recall for np.interp
        _, unique_idx = np.unique(rec_rev, return_index=True)
        precs.append(np.interp(REC_GRID, rec_rev[unique_idx], prec_rev[unique_idx]))
    if not precs:
        return REC_GRID, np.full_like(REC_GRID, PR_BASELINE)
    return REC_GRID, np.mean(precs, axis=0)


# ── Figure 1–5: Mean ROC and PR curves per classifier (one per encoding) ─────

def fig_roc_pr_per_model(preds, model_key, out_name):
    """Mean ROC and PR curves for one classifier across all encodings.

    Each curve is the average over CV folds 0-4. Individual fold curves are
    not plotted to keep the figure legible. Fold 5 (temporal) is omitted
    here and shown separately in fig_cv_vs_temporal.
    """
    # CV folds only (0-4)
    sub = preds[(preds['model'] == model_key) & (preds['fold'] < 5)]

    fig, axes = plt.subplots(1, 2, figsize=(26, 11))
    ax_roc, ax_pr = axes

    for enc in ENC_ORDER:
        colour = ENC_COLOURS[enc]
        ls     = ENC_LINESTYLES[enc]
        label  = ENC_LABELS[enc]
        enc_data = sub[sub['encoding'] == enc]
        if enc_data.empty:
            continue

        # Mean ROC across folds
        fpr_g, mean_tpr = _mean_roc(enc_data)
        ax_roc.plot(fpr_g, mean_tpr, color=colour, linestyle=ls,
                    linewidth=3, label=label)

        # Mean PR across folds
        rec_g, mean_prec = _mean_pr(enc_data)
        ax_pr.plot(rec_g, mean_prec, color=colour, linestyle=ls,
                   linewidth=3, label=label)

    # ROC diagonal baseline
    ax_roc.plot([0, 1], [0, 1], color='black', linestyle='--',
                linewidth=1.5, alpha=0.35)
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)

    # PR baseline (random classifier = class prevalence)
    ax_pr.axhline(PR_BASELINE, color='black', linestyle='--',
                  linewidth=1.5, alpha=0.35)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)

    # Shared legend below both panels
    handles, labels = ax_roc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.10), frameon=False)

    plt.tight_layout()
    _save(out_name)


# ── Figure 6: Encoding ablation for XGBoost ───────────────────────────────────

def fig_encoding_ablation(metrics):
    """Grouped bar chart: AUC-ROC and AUC-PR for XGBoost across encodings (CV mean)."""
    cv = metrics[(metrics['model'] == 'xgboost') & (metrics['fold'] < 5)]
    grp = cv.groupby('encoding')[['auc_roc', 'auc_pr']].mean().reset_index()
    grp['order'] = grp['encoding'].map({e: i for i, e in enumerate(ENC_ORDER)})
    grp = grp.sort_values('order')

    x     = np.arange(len(ENC_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(22, 11))
    # AUC-ROC: solid fill; AUC-PR: hatched to distinguish without extra colour
    ax.bar(x - width / 2, grp['auc_roc'].values, width,
           color='#2255AA', alpha=0.85, label='AUC-ROC')
    ax.bar(x + width / 2, grp['auc_pr'].values,  width,
           color='#CC4400', alpha=0.85, hatch='//', label='AUC-PR')

    ax.set_xlabel('Feature configuration')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels([ENC_LABELS[e] for e in ENC_ORDER], rotation=20, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save('encoding_ablation.pdf')


# ── Figure 7: CV vs temporal AUC-ROC per model ───────────────────────────────

def fig_cv_vs_temporal(metrics):
    """Grouped bar chart: CV mean AUC-ROC vs temporal fold AUC-ROC per classifier."""
    cv_vals, t5_vals = [], []
    for m in MODEL_ORDER:
        enc = MODEL_BEST_ENC[m]
        cv_val = metrics[(metrics['model'] == m) & (metrics['encoding'] == enc)
                         & (metrics['fold'] < 5)]['auc_roc'].mean()
        t5_row = metrics[(metrics['model'] == m) & (metrics['encoding'] == enc)
                         & (metrics['fold'] == 5)]
        t5_val = t5_row['auc_roc'].values[0] if not t5_row.empty else np.nan
        cv_vals.append(cv_val)
        t5_vals.append(t5_val)

    x     = np.arange(len(MODEL_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.bar(x - width / 2, cv_vals, width, color='#2255AA', alpha=0.85,
           label='Cross-validation (folds 0\u20134)')
    ax.bar(x + width / 2, t5_vals, width, color='#CC4400', alpha=0.85,
           hatch='//', label='Temporal fold (fold 5)')

    ax.set_xlabel('Classifier')
    ax.set_ylabel('AUC-ROC')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylim(0.60, 0.86)
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save('cv_vs_temporal.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    metrics = load_metrics()
    preds   = load_predictions()

    # Per-model mean ROC/PR figures (one per classifier)
    for model_key, out in [
        ('xgboost',      'task3_xgboost.pdf'),
        ('mlp',          'task3_mlp.pdf'),
        ('randomforest', 'task3_randomforest.pdf'),
        ('lightgbm',     'task3_lightgbm.pdf'),
        ('ridge',        'task3_ridge.pdf'),
    ]:
        fig_roc_pr_per_model(preds, model_key, out)

    fig_encoding_ablation(metrics)
    fig_cv_vs_temporal(metrics)

    print('All figures generated.')

---
title: surgical-procedure-deviation-detection
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🔍 Surgical Procedure Deviation Detection</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=Surgical+Deviation+Detection+%26+Duration+Prediction;BERT+%2B+Classical+NLP+%2B+Structured+Baselines;Binary+Classification+%7C+RAG-Augmented+%7C+5-Fold+CV" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-3b82f6?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient+Boosting-4f46e5?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🔍 Surgical Procedure Deviation Detection** — A dual-task framework combining surgical duration prediction (regression) with procedure deviation detection (binary classification), using pre-operative text and structured features with BERT embeddings, FAISS-based RAG retrieval, and Optuna-tuned ML models.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Pipeline Stages](#-pipeline-stages)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Outputs & Artifacts](#-outputs--artifacts)
- [Reproducibility](#-reproducibility)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🎯 <b>Dual-Task Design</b></td>
    <td>Task 1: surgical duration regression · Task 3: procedure deviation binary classification — both from pre-operative data only</td>
  </tr>
  <tr>
    <td>🤖 <b>BERT Text Embeddings</b></td>
    <td>ClinicalBERT and SentenceBERT encodings of <code>scheduled_procedure</code> text, PCA-compressed fold-wise to prevent leakage</td>
  </tr>
  <tr>
    <td>🔍 <b>RAG-Augmented Detection</b></td>
    <td>FAISS nearest-neighbour retrieval at inference time to flag high-risk deviation cases pre-operatively</td>
  </tr>
  <tr>
    <td>⚙️ <b>Optuna Hyperparameter Tuning</b></td>
    <td>TPE-based search with early stopping across Ridge, Lasso, ElasticNet, Random Forest, XGBoost, and MLP</td>
  </tr>
  <tr>
    <td>📊 <b>Comprehensive Metrics</b></td>
    <td>Regression: MAE, RMSE, SMAPE, R² · Classification: AUC-ROC, F1, precision, recall, balanced accuracy</td>
  </tr>
  <tr>
    <td>📄 <b>Publication-Ready Outputs</b></td>
    <td>LaTeX tables and PDF figures auto-exported to <code>results/</code> and <code>overleaf/</code></td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│           Surgical Procedure Deviation Detection                   │
│                                                                    │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │  Pre-op    │───▶│  Text Encoding  │───▶│  Task 1          │  │
│  │  Features  │    │  (BERT / TF-IDF)│    │  Duration (MAE)  │  │
│  └────────────┘    └────────┬────────┘    └──────────────────┘  │
│                             │                                      │
│                    ┌────────▼────────┐    ┌──────────────────┐   │
│                    │  FAISS Index    │───▶│  Task 3          │   │
│                    │  (RAG retrieval)│    │  Deviation (AUC) │   │
│                    └─────────────────┘    └──────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- `data/raw/casetime.csv` (source dataset — not included in repo)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/surgical-procedure-deviation-detection.git
cd surgical-procedure-deviation-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scipy scikit-learn optuna xgboost tensorflow torch \
            transformers sentence-transformers faiss-cpu matplotlib
```

### Run the Pipelines

```bash
# Task 1 — Duration Prediction (regression, 4-stage pipeline)
python code/pipeline.py

# Task 3 — Deviation Detection (binary classification + RAG)
python code/pipeline_task3.py

# Generate publication figures and LaTeX tables
python code/figures.py
```

Each pipeline auto-skips completed stages and prompts for model selection in Stage 04.

---

## 📊 Pipeline Stages

### Task 1 — Duration Prediction (`pipeline.py`)

| Stage | Description | Output |
|-------|-------------|--------|
| 01 Data Cleaning | Ingest CSV, validate types, persist Clean table | `data/surgical_data.db` |
| 02 BERT Encoding | ClinicalBERT & SentenceBERT embeddings, cached as `.npy` | `data/bert_cache/` |
| 03 Feature Engineering | 5-fold CV · impute · one-hot · fold-wise PCA | `data/fold_encoded.db` |
| 04 Model Training | Optuna HPO · Ridge/Lasso/RF/XGB/MLP → metrics & predictions | `results/result.db` |

### Task 3 — Deviation Detection (`pipeline_task3.py`)

| Stage | Description | Output |
|-------|-------------|--------|
| 01 Data Cleaning | Derive `deviation_label` from scheduled vs performed procedure | `data/task3_clean.db` |
| 02 BERT Encoding | Shared BERT cache from Task 1 (reused) | `data/bert_cache/` |
| 03 Feature Engineering | Fold-wise encoding + FAISS index construction (k=5) | `data/fold_encoded_task3.db` |
| 04 Model Training | All classifiers auto-run with class-weight balancing | `data/outputs/result_task3.db` |

---

## 🧠 ML Models

```python
# Task 1 — Regression
regressors = {
    "ridge":          "Ridge Regression (Optuna α)",
    "lasso":          "Lasso Regression (Optuna α)",
    "random_forest":  "RandomForestRegressor (Optuna)",
    "xgboost":        "XGBRegressor with early stopping (Optuna)",
    "mlp":            "TensorFlow MLP, AdamW, BatchNorm, depth 1–3 (Optuna)",
}

# Task 3 — Classification
classifiers = {
    "ridge":          "LogisticRegression L2 (Optuna C)",
    "lasso":          "LogisticRegression L1 (Optuna C)",
    "elasticnet":     "LogisticRegression ElasticNet (Optuna C, l1_ratio)",
    "random_forest":  "RandomForestClassifier class_weight='balanced' (Optuna)",
    "xgboost":        "XGBClassifier with scale_pos_weight (Optuna)",
    "mlp":            "TensorFlow MLP, class-weighted loss (Optuna)",
}
```

---

## 📁 Project Structure

```
surgical-procedure-deviation-detection/
│
├── 📂 code/
│   ├── 📄 pipeline.py           # Task 1: duration regression (Stages 01–04)
│   ├── 📄 pipeline_task3.py     # Task 3: deviation detection (Stages 01–04)
│   └── 📄 figures.py            # Figure and LaTeX table generation
│
├── 📂 data/
│   ├── 📂 raw/                  # Source data — read only, never modify
│   ├── 📂 bert_cache/           # Cached BERT embeddings (.npy) — shared across tasks
│   └── 📂 processed/            # Fold-encoded feature matrices and FAISS indexes
│
├── 📂 results/                  # Logs, PDF figures, and LaTeX tables
│
├── 📂 overleaf/                 # LaTeX manuscript
│   └── 📄 *.tex                 # Paper sections
│
├── 📂 flowchart/                # Pipeline flowchart assets
└── 📄 sync.ps1                  # Git sync utility
```

---

## 📦 Outputs & Artifacts

- **Databases** (local only, not in repo)
  - `data/surgical_data.db` — cleaned cases and fold indices (Task 1)
  - `data/task3_clean.db` — deviation-labelled dataset (Task 3)
  - `data/fold_encoded.db` — PCA-compressed feature matrices (Task 1)
  - `data/fold_encoded_task3.db` — feature matrices + FAISS indexes (Task 3)
  - `data/outputs/result_task3.db` — classification metrics, predictions, importances
- **Logs** — `results/*.log` per model and stage
- **Figures** — comparison PDFs in `results/`, publication-ready plots in `overleaf/`

---

## 🔁 Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) throughout all stages.
- All preprocessing (imputation, encoding, PCA) is fit on train folds only — no leakage.
- BERT embeddings are cached so repeated runs are fast and identical.
- Stages 01–03 auto-skip when outputs are present; Stage 04 resumes incomplete model runs.

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional medical advice of any kind. All datasets used are subject to institutional data-use agreements — no patient-identifiable information is included in this repository. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/surgical-procedure-deviation-detection?style=social)](https://github.com/mnoorchenar/surgical-procedure-deviation-detection)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/surgical-procedure-deviation-detection?style=social)](https://github.com/mnoorchenar/surgical-procedure-deviation-detection/fork)

<sub>This project is developed purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>

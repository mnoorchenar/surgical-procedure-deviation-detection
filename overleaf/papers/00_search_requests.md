# Search Requests — Keyword Log
# Maintained by Claude. Records all search clusters, their status, and internal rationale.
# The user sees only the INTRO SEARCH: / RW SEARCH: output lines — not this file.
# Status: PENDING = not yet searched | PROVIDED = user submitted candidates | DONE = written

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## STEP 5 — INTRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Target: 15–25 citations. Five clusters covering the 7-paragraph structure.
Place candidate PDFs in overleaf/papers/02_selected_papers_for_introduction/
Add BibTeX to 01_introduction.bib only after Stage 1 selection is confirmed.

Already in references.bib — do NOT duplicate in intro bib:
  lewis2020retrieval, akiba2019optuna, alsentzer2019publicly,
  reimers2019sentence, benjamini1995controlling, johnson2019billion

---

### Cluster A — Unplanned surgical procedure changes: incidence and clinical impact
# Supports P1 (real-world problem) and P2 (scale)
# STATUS: PENDING

INTRO SEARCH: "unplanned procedure change" OR "procedure deviation" operating room incidence
INTRO SEARCH: "surgical case cancellation" day-of causes patient safety outcomes
INTRO SEARCH: intraoperative "change of plan" surgical procedure unexpected conversion
INTRO SEARCH: "wrong-site surgery" OR "unintended procedure" adverse event operating room

---

### Cluster B — Operating room efficiency, scheduling disruption, and cost
# Supports P2 (economic/operational scale) and gap framing in P4
# STATUS: PENDING

INTRO SEARCH: "operating room" scheduling disruption cost efficiency utilisation
INTRO SEARCH: "OR scheduling" uncertainty prediction optimisation literature review
INTRO SEARCH: surgical schedule deviation downstream cost resource reallocation hospital
INTRO SEARCH: "operating room" efficiency machine learning prediction cancellation delay

---

### Cluster C — Machine learning for pre-operative surgical outcome prediction from EHR
# Supports P3 (current state of the art — structured data ML)
# STATUS: PENDING

INTRO SEARCH: "preoperative" OR "pre-operative" prediction machine learning "electronic health record"
INTRO SEARCH: surgical outcome prediction structured EHR XGBoost LightGBM gradient boosting
INTRO SEARCH: "perioperative risk" prediction neural network clinical data imbalanced
INTRO SEARCH: postoperative complication prediction administrative data logistic regression

---

### Cluster D — Clinical BERT and transformer models for medical text encoding
# Supports P5 (technology background: BERT, deployment barriers)
# STATUS: PENDING

INTRO SEARCH: ClinicalBERT "clinical notes" BERT pretraining EHR text representation
INTRO SEARCH: "sentence transformer" OR "sentence-BERT" semantic similarity medical text
INTRO SEARCH: BERT biomedical text classification procedure coding NLP
INTRO SEARCH: large language model clinical deployment inference cost latency edge

---

### Cluster E — Retrieval-augmented augmentation for clinical prediction
# Supports P6 (proposed solution: RAG-style augmentation, no prior domain-specific variant)
# STATUS: PENDING

INTRO SEARCH: "retrieval-augmented" clinical prediction patient similarity EHR
INTRO SEARCH: "case-based reasoning" OR "nearest neighbour" clinical decision support outcome
INTRO SEARCH: FAISS vector similarity search clinical patient record retrieval
INTRO SEARCH: historical similar case augmentation surgical prediction feature engineering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## STEP 6 — RELATED WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Target: ~35 papers across 4 clusters. ~10 candidates per cluster from user.
Place full-text PDFs of flagged papers in overleaf/papers/04_selected_papers_for_related_works/
Add BibTeX to 03_related_works.bib only after Stage 1 selection is confirmed.

---

### Cluster 1 — Surgical Procedure Deviation and OR Scheduling Prediction
# Core cluster — directly comparable work; expect most papers to need full text
# STATUS: PENDING

RW SEARCH: "procedure deviation" OR "procedure change" prediction preoperative machine learning
RW SEARCH: surgical cancellation prediction model EHR administrative data
RW SEARCH: "intraoperative" procedure conversion prediction preoperative features
RW SEARCH: operating room schedule adherence prediction deep learning
RW SEARCH: "surgical case" rescheduling cancellation reason classification

---

### Cluster 2 — Clinical NLP and BERT-based EHR Representation
# Positions use of ClinicalBERT + SentenceBERT on procedure name text
# STATUS: PENDING

RW SEARCH: ClinicalBERT "clinical BERT" medical text classification downstream task
RW SEARCH: BioBERT PubMedBERT biomedical language model pretraining fine-tuning
RW SEARCH: "sentence-BERT" OR "sentence transformer" clinical text similarity embedding
RW SEARCH: BERT EHR representation procedure code diagnosis classification
RW SEARCH: medical procedure name embedding NLP surgical coding

---

### Cluster 3 — Retrieval-Augmented and Instance-Based Clinical Prediction
# Positions the RAG augmentation component (FAISS + historical neighbour features)
# STATUS: PENDING

RW SEARCH: "retrieval-augmented generation" clinical healthcare prediction RAG
RW SEARCH: "case-based reasoning" clinical decision support EHR similarity retrieval
RW SEARCH: nearest neighbour instance-based patient outcome prediction hospital
RW SEARCH: FAISS approximate nearest neighbour medical record retrieval
RW SEARCH: "knowledge-augmented" OR "memory-augmented" clinical prediction transformer

---

### Cluster 4 — Imbalanced Classification in Surgical Outcome Prediction
# Positions class-imbalance handling (no SMOTE used; threshold + class weights + AUC-PR)
# STATUS: PENDING

RW SEARCH: class imbalance surgical outcome prediction SMOTE oversampling
RW SEARCH: "AUC-PR" OR "average precision" imbalanced clinical risk model evaluation
RW SEARCH: "precision-recall" threshold optimisation clinical decision model
RW SEARCH: class weight cost-sensitive learning perioperative prediction
RW SEARCH: imbalanced classification rare surgical complication deep learning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ALREADY IN references.bib — DO NOT DUPLICATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  lewis2020retrieval      ✓ NeurIPS 2020, vol 33, pp 9459–9474
  akiba2019optuna         ✓ KDD 2019, pp 2623–2631
  alsentzer2019publicly   ✓ Clinical NLP Workshop 2019, pp 72–78
  reimers2019sentence     ✓ EMNLP-IJCNLP 2019, pp 3982–3992
  benjamini1995controlling ✓ JRSS-B vol 57(1), pp 289–300, 1995
  johnson2019billion      ✓ IEEE Trans Big Data vol 7(3), pp 535–547, 2021

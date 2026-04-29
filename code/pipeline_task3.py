# =============================================================================
# pipeline_task3.py  --  Stages 01 . 02 . 03 . 04
#
# Task 3: Surgical Procedure Deviation Detection (binary classification).
# Predicts whether the performed surgical procedure will deviate from the
# scheduled one using pre-operative features only.
#
# Stages 01-03 check completion on every run and skip if already done.
# Stage 04 automatically runs ALL models -- no interactive prompts.
# Stage 04 auto-detects and skips already-completed combos (resume mode).
# All tuneable settings live in the CONFIG block below -- nowhere else.
#
# Separate from pipeline.py (Task 1): uses different DB paths, cache files,
# log names, and result DB to avoid conflicts.
# =============================================================================

# =============================================================================
# +==========================================================================+
# |  CONFIG  --  edit only this block                                        |
# +==========================================================================+
# =============================================================================

# -- Paths ---------------------------------------------------------------------
RAW_CSV    = './data/casetime.csv'
DB_PATH    = './data/task3_clean.db'
BERT_DIR   = './data/bert_cache'
ENCODED_DB = './data/fold_encoded_task3.db'
RESULT_DB  = './results/result_task3.db'
LOG_DIR    = './results'

# -- Column names --------------------------------------------------------------
TARGET    = 'deviation_label'
TEXT_COL  = 'scheduled_procedure'
DX_COL    = 'most_responsible_dx'

# Saved as raw strings in Clean; one-hot encoded fold-wise in Stage 03
CATEGORICAL_FOLD_COLS = ['case_service', 'surgical_location']

# NaN columns imputed fold-wise in Stage 03 (fit on train only -- no leakage)
IMPUTE_COLS  = ['age_at_discharge', 'avg_BMI', 'anesthetic_type']
IMPUTE_TYPES = ['regression',       'regression', 'classification']

# Dropped in Stage 03 -- intraoperative measurements known only after surgery
EXCLUDE_COLS = ['procedure_minutes', 'procedure_time', 'induction_time',
                'emergence_time',    'scheduled_duration', 'actual_casetime_minutes']

# -- Cross-validation ----------------------------------------------------------
N_SPLITS     = 5
RANDOM_STATE = 42

# -- Text feature dimensionality -----------------------------------------------
# 128-dim PCA sufficient for binary classification; reduces memory per fold.
FEATURES_PER_COL = [128]

# -- RAG parameters ------------------------------------------------------------
RAG_K              = 5    # k nearest neighbours for FAISS retrieval
CLINICAL_THRESHOLD = 0.20  # flag cases with P(deviation) > 20% for pre-op review

# -- Optuna general ------------------------------------------------------------
N_TRIALS                = 20
TUNE_SET_FRACTION       = 0.25
OPTUNA_N_STARTUP_TRIALS = 5
# Linear models (ridge/lasso/elasticnet) are slow with class_weight='balanced' on large
# datasets -- cap their Optuna subset and trial count to keep total runtime manageable.
LINEAR_OPTUNA_SUBSET    = 5000   # rows used per Optuna trial for linear models (None = all)
LINEAR_N_TRIALS         = 5     # Optuna trials for linear models (1-2 hyperparams only)

# -- Optuna search bounds: Ridge / Lasso (as LogisticRegression C) -------------
# Capped at 10 -- very high C causes slow convergence (many liblinear/lbfgs iterations)
OPTUNA_C_LOW  = 1e-3
OPTUNA_C_HIGH = 10.0

# -- Optuna search bounds: ElasticNet (LogisticRegression) --------------------
OPTUNA_EN_C_LOW       = 1e-3
OPTUNA_EN_C_HIGH      = 10.0
OPTUNA_EN_L1_RATIO_LOW  = 0.0
OPTUNA_EN_L1_RATIO_HIGH = 1.0

# -- Optuna search bounds: Random Forest --------------------------------------
OPTUNA_RF_N_EST_LOW              = 100
OPTUNA_RF_N_EST_HIGH             = 250
OPTUNA_RF_MAX_DEPTH_LOW          = 3
OPTUNA_RF_MAX_DEPTH_HIGH         = 10
OPTUNA_RF_MIN_SAMPLES_SPLIT_LOW  = 2
OPTUNA_RF_MIN_SAMPLES_SPLIT_HIGH = 10
OPTUNA_RF_MIN_SAMPLES_LEAF_LOW   = 1
OPTUNA_RF_MIN_SAMPLES_LEAF_HIGH  = 5
OPTUNA_RF_MAX_FEATURES           = ['sqrt', 'log2', 0.3, 0.5, 0.7]

# -- Optuna search bounds: XGBoost --------------------------------------------
OPTUNA_XGB_N_EST_LOW        = 100
OPTUNA_XGB_N_EST_HIGH       = 500
OPTUNA_XGB_LR_LOW           = 0.01
OPTUNA_XGB_LR_HIGH          = 0.3
OPTUNA_XGB_MAX_DEPTH_LOW    = 3
OPTUNA_XGB_MAX_DEPTH_HIGH   = 8
OPTUNA_XGB_SUBSAMPLE_LOW    = 0.6
OPTUNA_XGB_SUBSAMPLE_HIGH   = 1.0
OPTUNA_XGB_COLSAMPLE_LOW    = 0.6
OPTUNA_XGB_COLSAMPLE_HIGH   = 1.0
OPTUNA_XGB_REG_ALPHA_LOW    = 1e-4
OPTUNA_XGB_REG_ALPHA_HIGH   = 10.0
OPTUNA_XGB_REG_LAMBDA_LOW   = 1e-4
OPTUNA_XGB_REG_LAMBDA_HIGH  = 10.0

# -- Optuna search bounds: LightGBM -------------------------------------------
OPTUNA_LGB_N_EST_LOW        = 100
OPTUNA_LGB_N_EST_HIGH       = 500
OPTUNA_LGB_LR_LOW           = 0.01
OPTUNA_LGB_LR_HIGH          = 0.3
OPTUNA_LGB_MAX_DEPTH_LOW    = 3
OPTUNA_LGB_MAX_DEPTH_HIGH   = 8
OPTUNA_LGB_NUM_LEAVES_LOW   = 15
OPTUNA_LGB_NUM_LEAVES_HIGH  = 127
OPTUNA_LGB_SUBSAMPLE_LOW    = 0.6
OPTUNA_LGB_SUBSAMPLE_HIGH   = 1.0
OPTUNA_LGB_COLSAMPLE_LOW    = 0.6
OPTUNA_LGB_COLSAMPLE_HIGH   = 1.0
OPTUNA_LGB_REG_ALPHA_LOW    = 1e-4
OPTUNA_LGB_REG_ALPHA_HIGH   = 10.0
OPTUNA_LGB_REG_LAMBDA_LOW   = 1e-4
OPTUNA_LGB_REG_LAMBDA_HIGH  = 10.0
LGB_EARLY_STOPPING_ROUNDS   = 20

# -- Optuna search bounds: MLP ------------------------------------------------
OPTUNA_MLP_N_LAYERS_LOW      = 1
OPTUNA_MLP_N_LAYERS_HIGH     = 3
OPTUNA_MLP_UNITS1_LOW        = 32
OPTUNA_MLP_UNITS1_HIGH       = 256
OPTUNA_MLP_UNITS2_LOW        = 16
OPTUNA_MLP_UNITS2_HIGH       = 128
OPTUNA_MLP_UNITS3_LOW        = 8
OPTUNA_MLP_UNITS3_HIGH       = 64
OPTUNA_MLP_DROPOUT_LOW       = 0.0
OPTUNA_MLP_DROPOUT_HIGH      = 0.5
OPTUNA_MLP_LR_LOW            = 1e-4
OPTUNA_MLP_LR_HIGH           = 1e-2
OPTUNA_MLP_WEIGHT_DECAY_LOW  = 1e-6
OPTUNA_MLP_WEIGHT_DECAY_HIGH = 1e-2
OPTUNA_MLP_ACTIVATIONS       = ['relu', 'elu', 'tanh']

# -- MLP fixed training settings -----------------------------------------------
MLP_EPOCHS_FINAL      = 200
MLP_EPOCHS_OPTUNA     = 30
MLP_BATCH_SIZE        = 512
MLP_PATIENCE_ES       = 15
MLP_PATIENCE_LR       = 5
MLP_LR_DECAY_FACTOR   = 0.5
MLP_MIN_LR            = 1e-6
MLP_CLIPNORM          = 1.0
MLP_OPTUNA_SUBSET_SIZE = 5000  # max rows per Optuna trial (None = all)

# -- XGBoost training settings -------------------------------------------------
XGB_TREE_METHOD           = 'hist'
XGB_DEVICE                = 'cuda'   # runtime-checked in Stage 04; falls back to 'cpu'
XGB_EARLY_STOPPING_ROUNDS = 20

# -- BERT model identifiers ----------------------------------------------------
CLINICALBERT_MODEL_ID = 'emilyalsentzer/Bio_ClinicalBERT'
SENTENCEBERT_MODEL_ID = 'all-MiniLM-L6-v2'
BERT_BATCH_CLINICAL   = 32
BERT_BATCH_SENTENCE   = 64
BERT_MAX_LENGTH       = 64

# -- Models to run -------------------------------------------------------------
ALL_MODELS = ['ridge', 'randomforest', 'xgboost', 'lightgbm', 'mlp']

# -- DB table names ------------------------------------------------------------
CLEAN_TABLE = 'Task3Clean'
FOLD_TABLE  = 'fold_indices'

# -- BERT cache task map (task3-prefixed to avoid conflict with pipeline.py) ---
S02_TASKS = {
    1: ('clinicalbert', f'task3_clinicalbert_{TEXT_COL}.npy'),
    2: ('sentencebert', f'task3_sentencebert_{TEXT_COL}.npy'),
    # tinybert requires distill_bert.py to be run first; omitted if model files absent
    # 3: ('tinybert',   f'task3_tinybert_{TEXT_COL}.npy'),
}

# =============================================================================
# IMPORTS
# =============================================================================
import os, sys, re, sqlite3, time, warnings
import numpy as np
import pandas as pd

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs('./data', exist_ok=True)
os.makedirs(BERT_DIR, exist_ok=True)
os.makedirs('./results', exist_ok=True)

warnings.filterwarnings('ignore')

# =============================================================================
# SHARED UTILITIES
# =============================================================================

class _Tee:
    def __init__(self, path):
        self._t = sys.stdout
        self._f = open(path, 'w', encoding='utf-8', buffering=1)
    def write(self, m):  self._t.write(m); self._f.write(m)
    def flush(self):     self._t.flush();  self._f.flush()
    def close(self):     sys.stdout = self._t; self._f.close()

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

def _print_missing(df, label=''):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print(f"  No missing values{' -- ' + label if label else ''}."); return
    total = len(df)
    print(f"\n  Missing -- {label}:")
    print(f"  {'Column':<45} {'N':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for col, cnt in missing.items():
        print(f"  {col:<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def _print_freq(series, label='', top_n=20):
    vc    = series.value_counts(dropna=False)
    total = len(series)
    print(f"\n  Freq -- {label or series.name}  (top {top_n}):")
    print(f"  {'Value':<45} {'N':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for val, cnt in vc.head(top_n).items():
        print(f"  {str(val):<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def _print_numeric(df, cols, label=''):
    print(f"\n  Numeric summary -- {label}:")
    print(f"  {'Column':<35} {'N':>8} {'Mean':>9} {'SD':>9} {'Min':>9} {'Median':>9} {'Max':>9} {'NaN':>6}")
    print(f"  {'-'*97}")
    for col in cols:
        if col not in df.columns: continue
        s = df[col].dropna()
        if len(s) == 0: continue
        print(f"  {col:<35} {len(s):>8,} {s.mean():>9.2f} {s.std():>9.2f} {s.min():>9.2f} {s.median():>9.2f} {s.max():>9.2f} {df[col].isna().sum():>6,}")

# -- Stage completion checks ---------------------------------------------------

def _s01_is_done():
    if not os.path.exists(DB_PATH): return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if CLEAN_TABLE not in tables or FOLD_TABLE not in tables: return False
            return (conn.execute(f"SELECT COUNT(*) FROM {CLEAN_TABLE}").fetchone()[0] > 0 and
                    conn.execute(f"SELECT COUNT(*) FROM {FOLD_TABLE}").fetchone()[0] > 0)
    except: return False

def _s02_task_is_done(task_id):
    _, fname = S02_TASKS[task_id]
    return os.path.exists(os.path.join(BERT_DIR, fname))

def _s03_expected_count():
    """Expected encoded_matrices row count given whichever BERT caches exist."""
    n_bert_available = sum(1 for tid in S02_TASKS if _s02_task_is_done(tid))
    # Per fold, per split (train+val=2):
    #   1 struct_only
    #   n_bert_available * len(FEATURES_PER_COL) BERT encodings
    #   n_bert_available * len(FEATURES_PER_COL) RAG encodings
    # Folds 0-4 (N_SPLITS) + fold 5 (temporal) = N_SPLITS+1 folds
    n_folds = N_SPLITS + 1  # includes temporal fold
    per_fold = 2 * (1 + n_bert_available * len(FEATURES_PER_COL) * 2)
    return n_folds * per_fold

def _s03_is_done():
    if not os.path.exists(ENCODED_DB): return False
    try:
        with sqlite3.connect(ENCODED_DB) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'encoded_matrices' not in tables: return False
            n = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            return n >= _s03_expected_count()
    except: return False

# =============================================================================
# STAGE 01 -- PRE-PROCESSING
# =============================================================================

def run_stage01():
    if _s01_is_done():
        print(f"  [>>]  Stage 01 already complete ({DB_PATH} -> {CLEAN_TABLE} + {FOLD_TABLE}). Skipping.")
        return

    tee = _Tee(f'{LOG_DIR}/task3_01_preprocessing.log')
    sys.stdout = tee
    try:
        sep("STAGE 01 -- PRE-PROCESSING  (Task 3: Procedure Deviation Detection)")

        sep("1 -- LOAD & INITIAL CLEANING")
        df = pd.read_csv(RAW_CSV)
        df.columns = df.columns.str.strip()
        n_init = len(df)
        print(f"  Loaded : {n_init:,} rows x {df.shape[1]} columns")
        print(f"  Columns: {df.columns.tolist()}")
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace(r'^\s*$', np.nan, regex=True)
        pat = re.compile(r'^\s*(nan|none|null|na|n/a|missing|unknown|\?|-|)\s*$', re.IGNORECASE)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and pat.match(x) else x)
        print(f"  Shape after normalize + clean_missing: {df.shape[0]:,} x {df.shape[1]}")
        _print_missing(df, "after initial cleaning")

        sep("2 -- DEVIATION LABEL DERIVATION")
        # Drop rows where either scheduled_procedure or procedure is NaN
        # (cannot compute deviation label without both values)
        before = len(df)
        df = df[df['scheduled_procedure'].notna() & df['procedure'].notna()].copy()
        print(f"  Dropped {before - len(df):,} rows with NaN scheduled_procedure or procedure")
        # Derive binary deviation label (string comparison after strip/lower already done above)
        df[TARGET] = (df['scheduled_procedure'] != df['procedure']).astype(int)
        n_dev   = df[TARGET].sum()
        n_total = len(df)
        print(f"\n  Deviation label distribution:")
        print(f"    Label=0 (no deviation)  : {n_total - n_dev:>10,}  ({(n_total - n_dev)/n_total*100:.2f}%)")
        print(f"    Label=1 (deviation)      : {n_dev:>10,}  ({n_dev/n_total*100:.2f}%)")
        print(f"    Class imbalance ratio    : {(n_total - n_dev) / max(n_dev, 1):.1f}:1 (neg:pos)")

        sep("3 -- DATETIME FEATURES")
        # Task 3 ONLY requires pre-operative datetimes (scheduled_start_dttm).
        # Do NOT filter on intraoperative datetimes (procedure_start_dttm,
        # procedure_stop_dttm, procedure_minutes, etc.) because deviation cases
        # have NaN in those columns -- filtering on them would drop ALL deviations.
        preop_dt_cols = ['scheduled_start_dttm']
        before = len(df)
        df.dropna(subset=[c for c in preop_dt_cols if c in df.columns], inplace=True)
        print(f"  Removed {before - len(df):,} rows with missing scheduled_start_dttm -> {len(df):,} remaining")
        print(f"  NOTE: intraoperative datetimes (procedure_*_dttm, procedure_minutes, etc.)")
        print(f"        NOT used as row filters -- deviation cases have NaN there by design.")

        # Parse only the pre-operative datetime columns we need
        parse_cols = [c for c in df.columns if any(x in c.lower() for x in ['dttm', 'date'])]
        for c in parse_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')

        # Keep _sort_dttm for temporal split (copy before dropping parse_cols)
        df['_sort_dttm'] = df['scheduled_start_dttm'].copy()

        # Derive pre-operative time-based features
        df['scheduled_start_hour'] = df['scheduled_start_dttm'].dt.hour
        df['month_of_year']        = df['scheduled_start_dttm'].dt.month
        df['day_of_week']          = df['scheduled_start_dttm'].dt.dayofweek

        # Drop all datetime parse columns (including intraoperative ones)
        df.drop(columns=parse_cols, inplace=True)
        print(f"  Derived: scheduled_start_hour, month_of_year, day_of_week")
        print(f"  Kept   : _sort_dttm  (for temporal split; dropped before DB save)")
        print(f"  Target : {TARGET}  (binary: scheduled != performed procedure)")

        sep("4 -- IMPLAUSIBLE AGE / BMI FILTER  (hardcoded bounds)")
        inv_age = ~df['age_at_discharge'].between(18, 130)
        df.loc[inv_age, 'age_at_discharge'] = np.nan
        print(f"  age_at_discharge: clamped {inv_age.sum():,} outside [18,130] -> NaN  (fold-wise imputation)")
        inv_bmi = ~df['avg_BMI'].between(5, 200)
        df.loc[inv_bmi, 'avg_BMI'] = np.nan
        print(f"  avg_BMI:          clamped {inv_bmi.sum():,} outside [5,200]   -> NaN  (fold-wise imputation)")

        sep("5 -- DROP IDENTIFIER COLUMNS")
        drop_ids = [c for c in ['patient_id', 'avg_wt_enct', 'avg_ht_enct', 'week_day'] if c in df.columns]
        df.drop(columns=drop_ids, inplace=True)
        print(f"  Dropped: {drop_ids}  ->  shape={df.shape[0]:,} x {df.shape[1]}")
        print(f"  Retained: case_id  (join key for post-hoc subgroup analysis -- not used as a feature)")

        sep("6 -- DROP INTRAOPERATIVE / TARGET-LEAKING COLUMNS")
        # Drop: procedure (ground truth), operative_dx (filled intraop),
        # actual_casetime_minutes, procedure_minutes (intraop), and any others
        # that survive from parse_cols above (computed time columns)
        intraop_drop = []
        for c in ['procedure', 'operative_dx', 'actual_casetime_minutes',
                  'procedure_minutes', 'procedure_time', 'induction_time',
                  'emergence_time', 'scheduled_duration', 'or_entry_hour']:
            if c in df.columns:
                intraop_drop.append(c)
        df.drop(columns=intraop_drop, inplace=True)
        print(f"  Dropped intraoperative columns: {intraop_drop}")
        print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")

        sep("7 -- MISSINGNESS")
        _print_missing(df, "before dropping non-imputable")
        # must_have: columns that must be present (cannot impute, no useful signal if missing)
        must_have = ['ASA_score', 'sex', 'surg_encounter_type', 'case_service', 'scheduled_procedure']
        before = len(df); df.dropna(subset=must_have, inplace=True)
        print(f"\n  Dropped {before - len(df):,} rows missing in {must_have}")
        _print_missing(df, "after dropping non-imputable")

        sep("8 -- CATEGORICAL CLEANING")

        df['ASA_score'] = df['ASA_score'].apply(
            lambda x: int(m.group(1)) if (m := re.match(r'^([1-5])(?:e)?$', str(x).strip().lower())) else np.nan
        )
        df.dropna(subset=['ASA_score'], inplace=True)
        _print_freq(df['ASA_score'], "ASA_score after cleanup")

        df['OR_trip_sequence'] = (df['OR_trip_sequence'] == 1).astype(int)
        for col, val in [
            ('first_scheduled_case_of_day_status', 'first scheduled case of day'),
            ('last_scheduled_case_of_day_status',  'last scheduled case of day'),
            ('primary_procedure_status',           'primary procedure'),
        ]:
            df[col] = (df[col] == val).astype(int)

        before = len(df)
        df = df[df['sex'].isin(['male', 'female'])].copy()
        print(f"\n  [Sex]  dropped {before - len(df):,} rows not in ['male','female']  (hardcoded rule)")
        df['sex'] = (df['sex'] == 'male').astype(int)

        df['surgery_encounter_inpatient'] = np.where(
            df['surg_encounter_type'].str.lower().isin(['same day admission', 'one day stay']), 0,
            np.where(df['surg_encounter_type'].str.lower() == 'inpatient', 1, np.nan)
        )
        df.drop(columns=['surg_encounter_type'], inplace=True)
        before = len(df)
        df = df[df['surgery_encounter_inpatient'].notna()].copy()
        print(f"\n  [Surgical Encounter]  mapped: 0=outpatient  1=inpatient")
        print(f"  Dropped {before - len(df):,} rows with unknown encounter type  (hardcoded rule)")

        def _simplify_loc(loc):
            loc = str(loc).strip().lower()
            if loc.startswith('vh or'):   return 'VH_OR'
            if loc.startswith('uh or'):   return 'UH_OR'
            if loc.startswith('vsc or'):  return 'VSC_OR'
            if loc.startswith('zzvh ob'): return 'OB_VH'
            if 'anesthesia' in loc:       return 'Anesthesia'
            if any(x in loc for x in ['pacu', 'pmdu', 'phase', 'recovery']): return 'Recovery'
            if any(x in loc for x in ['tee', 'pain']): return 'Procedure_Room'
            if 'alternate' in loc:        return 'Alternate_OR'
            return 'Other'
        df['surgical_location'] = df['surgical_location'].apply(_simplify_loc)
        before = len(df)
        df = df[df['surgical_location'] != 'Other'].copy()
        print(f"\n  [Surgical Location]  dropped {before - len(df):,} rows mapped to 'Other'  (hardcoded rule)")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['surgical_location'], "surgical_location after mapping")

        svc_map = {
            'orthopedic surgery':    'Orthopedic',       'general surgery':      'General_Surgery',
            'obstetrics/gynecology': 'OB_GYN',           'otolaryngology':       'ENT',
            'urology':               'Urology',          'plastic surgery':      'Plastic_Surgery',
            'neurosurgery':          'Neurosurgery',     'cardiac surgery':      'Cardiac_Surgery',
            'vascular surgery':      'Vascular_Surgery', 'thoracic surgery':     'Thoracic_Surgery',
            'dental surgery':        'Dental_Surgery',   'ophthalmology':        'Ophthalmology',
            'lrcp surg':             'Surgical_Oncology','cardiology surg':      'Cardiac_Surgery',
            'medicine surg':         np.nan,             'unknown case service': np.nan,
            'anesthesia surg':       np.nan,
        }
        df['case_service'] = df['case_service'].str.lower().map(svc_map)
        before = len(df); df.dropna(subset=['case_service'], inplace=True)
        print(f"\n  [Case Service]  dropped {before - len(df):,} unmapped rows  (hardcoded rule)")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['case_service'], "case_service after mapping")

        df['anesthetic_type'] = df['anesthetic_type'].str.replace(r'^general/|/general', '', regex=True).str.strip()
        anesthesia_map = {
            "general": "General",                "general/epidural": "Combined",
            "general/regional": "Combined",      "general/spinal": "Combined",
            "general/spinal opioid": "Combined",  "general/axillary": "Combined",
            "general rectal": "General",          "general endo": "General",
            "general/home regional": "Combined",  "spinal block": "Neuraxial",
            "epidural block": "Neuraxial",        "combined spinal/epidural": "Neuraxial",
            "lumbar epidural block": "Neuraxial", "brachial plexus block": "Regional",
            "supraclavicular block": "Regional",  "interscalene block": "Regional",
            "infraclavicular block": "Regional",  "intercostal brachial": "Regional",
            "sciatic catheter block": "Regional", "paravertebral nerve block": "Regional",
            "transverse abdominus plane block": "Regional", "lumbar plexus block": "Regional",
            "cervical plexus block": "Regional",  "ilioinguinal block": "Regional",
            "axillary block": "Regional",         "femoral block": "Regional",
            "popliteal block": "Regional",        "saphenous knee block": "Regional",
            "saphenous elbow block": "Regional",  "suprascapular block": "Regional",
            "home regional": "Regional",          "regional": "Regional",
            "regional/home regional": "Regional", "local": "Local",
            "local with standby": "Local",        "local/sedation": "Local",
            "local - monitored anesthesia care": "Local", "facial block": "Local",
            "o'brien block": "Local",             "peribulbar and retrobulbar block": "Local",
            "ankle block": "Local",               "caudal block": "Local",
            "bier block": "Local",                "local neurolept": "Sedation",
            "iv sedation": "Sedation",            "neurolept": "Sedation",
            "iv regional": "Sedation",            "no anesthesia given": np.nan,
            "system": np.nan,                     "other": np.nan,
        }
        df['anesthetic_type'] = df['anesthetic_type'].map(anesthesia_map)
        print(f"\n  [Anesthetic Type]  NaN kept for fold-wise imputation")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['anesthetic_type'], "anesthetic_type after mapping")

        sep("9 -- TEXT COLUMNS")
        print(f"  Retained: {TEXT_COL}  (BERT-encoded fold-wise in Stage 03)")
        print(f"  Retained: {DX_COL}  (ICD-10 chapter-encoded fold-wise in Stage 03)")
        print(f"\n  Sample {TEXT_COL} values (first 5 rows):")
        for i in range(min(5, len(df))):
            print(f"    Row {i}: {str(df[TEXT_COL].iloc[i])[:120]}")
        print(f"\n  Sample {DX_COL} values (first 5 rows):")
        for i in range(min(5, len(df))):
            print(f"    Row {i}: {str(df[DX_COL].iloc[i])[:60]}")

        sep("10 -- DEVIATION LABEL FINAL DISTRIBUTION (after all cleaning)")
        df.reset_index(drop=True, inplace=True)
        n_dev   = df[TARGET].sum()
        n_total = len(df)
        print(f"\n  After cleaning:")
        print(f"    Total rows            : {n_total:,}")
        print(f"    Label=0 (no deviation): {n_total - n_dev:>10,}  ({(n_total - n_dev)/n_total*100:.2f}%)")
        print(f"    Label=1 (deviation)   : {n_dev:>10,}  ({n_dev/n_total*100:.2f}%)")
        print(f"    Imbalance ratio (neg:pos) : {(n_total - n_dev) / max(n_dev, 1):.1f}:1")
        print(f"    Optimal XGB scale_pos_weight : {(n_total - n_dev) / max(n_dev, 1):.2f}")

        sep("11 -- FINAL DATASET SUMMARY")
        text_c   = [TEXT_COL, DX_COL]
        cat_c    = [c for c in df.columns if c in CATEGORICAL_FOLD_COLS + ['anesthetic_type']]
        struct_c = [c for c in df.columns if c not in text_c + cat_c + [TARGET, 'case_id', '_sort_dttm']]
        print(f"\n  Rows          : {len(df):,}  (removed {n_init - len(df):,} = {(n_init-len(df))/n_init*100:.1f}%)")
        print(f"  Columns       : {df.shape[1]}")
        print(f"  Target        : {TARGET}  (binary deviation label)")
        print(f"  Text columns  : {text_c}  (encoded in Stage 03)")
        print(f"  Fold one-hot  : {cat_c}  (one-hot in Stage 03, train categories only)")
        print(f"  Structured    : {len(struct_c)}  {struct_c}")
        _print_missing(df, "final Clean table")

        sep("SAVE + FOLD INDEX GENERATION")
        # Save to DB (without _sort_dttm -- used only for fold generation below)
        df_save = df.drop(columns=['_sort_dttm']).copy()
        with sqlite3.connect(DB_PATH) as conn:
            df_save.to_sql(CLEAN_TABLE, conn, if_exists='replace', index=False)
        print(f"  Saved '{CLEAN_TABLE}' -> {DB_PATH}  shape={df_save.shape}")

        # -- KFold (folds 0-4) + temporal split (fold=5) -----------------------
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_rows = []

        # Folds 0-4: standard KFold
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
            for idx in tr_idx:
                fold_rows.append({'fold': int(fold), 'split': 'train', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
            for idx in va_idx:
                fold_rows.append({'fold': int(fold), 'split': 'val',   'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})

        # Fold 5: temporal split (earliest 80% train, latest 20% val)
        sort_order = df['_sort_dttm'].argsort().values  # ascending by date
        n_temporal = len(sort_order)
        cutoff     = int(n_temporal * 0.80)
        tr_temp    = sort_order[:cutoff]
        va_temp    = sort_order[cutoff:]
        for idx in tr_temp:
            fold_rows.append({'fold': 5, 'split': 'train', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
        for idx in va_temp:
            fold_rows.append({'fold': 5, 'split': 'val',   'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})

        fold_df = pd.DataFrame(fold_rows)
        with sqlite3.connect(DB_PATH) as conn:
            fold_df.to_sql(FOLD_TABLE, conn, if_exists='replace', index=False)

        print(f"  KFold: n_splits={N_SPLITS}  shuffle=True  random_state={RANDOM_STATE}")
        print(f"  Temporal fold 5: train=earliest 80%, val=latest 20% by scheduled_start_dttm")
        print(f"\n  {'Fold':<6} {'Train':>12} {'Val':>12}  {'Note'}")
        print(f"  {'-'*50}")
        for fold in range(N_SPLITS):
            nt = len(fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')])
            nv = len(fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')])
            print(f"  {fold:<6} {nt:>12,} {nv:>12,}")
        nt5 = len(fold_df[(fold_df['fold']==5) & (fold_df['split']=='train')])
        nv5 = len(fold_df[(fold_df['fold']==5) & (fold_df['split']=='val')])
        print(f"  {'5':<6} {nt5:>12,} {nv5:>12,}  temporal split")

        print(f"\n  [OK] Stage 01 complete.  Log -> {LOG_DIR}/task3_01_preprocessing.log")

    finally:
        tee.close()


# =============================================================================
# STAGE 02 -- BERT CACHE
# =============================================================================

def _s02_compute_clinicalbert(texts):
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {CLINICALBERT_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_ID)
    model     = AutoModel.from_pretrained(CLINICALBERT_MODEL_ID)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device); print(f"  Device: {device}")
    embs      = []
    n_batches = (len(texts) + BERT_BATCH_CLINICAL - 1) // BERT_BATCH_CLINICAL
    for i in range(0, len(texts), BERT_BATCH_CLINICAL):
        batch = texts[i:i+BERT_BATCH_CLINICAL]
        with torch.no_grad():
            inp = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=BERT_MAX_LENGTH)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        if (i // BERT_BATCH_CLINICAL) % 50 == 0:
            print(f"    Batch {i//BERT_BATCH_CLINICAL+1}/{n_batches}  ({i:,}/{len(texts):,})")
    return np.vstack(embs)

def _s02_compute_sentencebert(texts):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {SENTENCEBERT_MODEL_ID}...")
    model = SentenceTransformer(SENTENCEBERT_MODEL_ID)
    return model.encode(texts, show_progress_bar=True, batch_size=BERT_BATCH_SENTENCE)

def _s02_compute_tinybert(texts):
    """Use the distilled TinySurgicalBERT model (run distill_bert.py first)."""
    import json as _json
    import torch as _torch
    from tokenizers import Tokenizer as _Tokenizer
    import torch.nn as _nn

    TINYBERT_DIR = './models/surgical_tiny_bert'
    tok_path = os.path.join(TINYBERT_DIR, 'tokenizer.json')
    cfg_path = os.path.join(TINYBERT_DIR, 'config.json')
    pt_path  = os.path.join(TINYBERT_DIR, 'pytorch_model.pt')

    for p in [tok_path, cfg_path, pt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"TinySurgicalBERT file missing: {p}\n"
                f"  Run:  python distill_bert.py  to generate it first."
            )

    cfg = _json.load(open(cfg_path))
    print(f"  Loading TinySurgicalBERT from {TINYBERT_DIR}  "
          f"(params={cfg.get('n_params', '?'):,}  output_dim={cfg['output_dim']})")

    tok = _Tokenizer.from_file(tok_path)
    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=cfg['max_seq_len'])
    tok.enable_truncation(max_length=cfg['max_seq_len'])

    class _TinyBERT(_nn.Module):
        def __init__(self, c):
            super().__init__()
            self.token_emb  = _nn.Embedding(c['vocab_size'], c['d_model'], padding_idx=0)
            self.pos_emb    = _nn.Embedding(c['max_seq_len'], c['d_model'])
            self.embed_norm = _nn.LayerNorm(c['d_model'])
            self.embed_drop = _nn.Dropout(c['dropout'])
            enc_layer = _nn.TransformerEncoderLayer(
                d_model=c['d_model'], nhead=c['nhead'],
                dim_feedforward=c['dim_feedforward'],
                dropout=c['dropout'], batch_first=True,
                norm_first=True, activation='gelu')
            self.encoder    = _nn.TransformerEncoder(enc_layer, num_layers=c['num_layers'])
            self.final_norm = _nn.LayerNorm(c['d_model'])
            self.output_proj= _nn.Linear(c['d_model'], c['output_dim'])
        def forward(self, input_ids, attention_mask):
            pos = _torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            x = self.embed_drop(self.embed_norm(x))
            x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
            return self.output_proj(self.final_norm(x)[:, 0, :])

    model = _TinyBERT(cfg)
    model.load_state_dict(_torch.load(pt_path, map_location='cpu', weights_only=True))
    model.eval()
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"  Device: {device}")

    batch_size = 512
    all_emb = []
    n = len(texts)
    lowers = [t.lower().strip() for t in texts]
    for i in range(0, n, batch_size):
        batch = lowers[i:i+batch_size]
        encs  = tok.encode_batch(batch)
        ids   = _torch.tensor([e.ids for e in encs], dtype=_torch.long).to(device)
        mask  = _torch.tensor([e.attention_mask for e in encs], dtype=_torch.long).to(device)
        with _torch.no_grad():
            all_emb.append(model(ids, mask).cpu().numpy())
        if i % 10240 == 0:
            print(f"    {i:,}/{n:,}")
    return np.vstack(all_emb).astype(np.float32)

def _s02_run_task(task_id):
    method, out_filename = S02_TASKS[task_id]
    out_path = os.path.join(BERT_DIR, out_filename)
    log_path = os.path.join(LOG_DIR, f"task3_02_bert_task{task_id}_{method}.log")
    tee = _Tee(log_path); sys.stdout = tee
    try:
        sep(f"TASK {task_id} -- {method.upper()} on '{TEXT_COL}'  (Task 3)")
        print(f"  Output : {out_path}")
        if os.path.exists(out_path):
            os.remove(out_path); print(f"  Removed existing cache -- recomputing fresh.")
        with sqlite3.connect(DB_PATH) as conn:
            df_txt = pd.read_sql(f"SELECT [{TEXT_COL}] FROM {CLEAN_TABLE}", conn)
        texts = df_txt[TEXT_COL].astype(str).tolist()
        print(f"  Loaded {len(texts):,} texts from '{TEXT_COL}'")
        t0 = time.time()
        if method == 'clinicalbert':
            embs = _s02_compute_clinicalbert(texts)
        elif method == 'sentencebert':
            embs = _s02_compute_sentencebert(texts)
        elif method == 'tinybert':
            embs = _s02_compute_tinybert(texts)
        else:
            raise ValueError(f"Unknown BERT method: {method}")
        print(f"\n  Embedding shape : {embs.shape}")
        print(f"  Elapsed         : {(time.time()-t0)/60:.2f} min")
        np.save(out_path, embs)
        print(f"  Saved -> {out_path}")
        print(f"  NOTE: Full {embs.shape[1]}-d embeddings stored. PCA applied per n in FEATURES_PER_COL={FEATURES_PER_COL} fold-wise in Stage 03.")
    finally:
        tee.close()

def run_stage02():
    sep("STAGE 02 -- BERT CACHE  (Task 3)")
    if all(_s02_task_is_done(tid) for tid in S02_TASKS):
        print(f"  [>>]  All BERT cache files already exist -- skipping Stage 02.")
        for tid, (method, fname) in S02_TASKS.items():
            arr = np.load(os.path.join(BERT_DIR, fname))
            print(f"    Task {tid}  {fname:<60}  shape={arr.shape}")
        return
    print()
    missing = [tid for tid in S02_TASKS if not _s02_task_is_done(tid)]
    print(f"  Running all missing BERT tasks automatically: {missing}")
    for tid in missing:
        _, fname = S02_TASKS[tid]
        print(f"  [{tid}]  {S02_TASKS[tid][0]:<16}  {fname}  [RUNNING]")
    print()
    for tid in missing:
        _s02_run_task(tid)
    sep("STAGE 02 COMPLETE")
    for tid in missing:
        _, fname = S02_TASKS[tid]
        arr = np.load(os.path.join(BERT_DIR, fname))
        print(f"  Task {tid}  {fname}  shape={arr.shape}")


# =============================================================================
# STAGE 03 -- FOLD ENCODING
# =============================================================================

def _s03_init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_matrices (
            fold       INTEGER,
            split      TEXT,
            encoding   TEXT,
            n_features INTEGER,
            rows       INTEGER,
            cols       INTEGER,
            dtype      TEXT,
            data       BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_targets (
            fold  INTEGER,
            split TEXT,
            rows  INTEGER,
            dtype TEXT,
            data  BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enc ON encoded_matrices (fold, split, encoding, n_features)")
    conn.commit()

def _s03_save_matrix(conn, fold, split, encoding, n_features, arr):
    conn.execute(
        "INSERT INTO encoded_matrices (fold,split,encoding,n_features,rows,cols,dtype,data) VALUES (?,?,?,?,?,?,?,?)",
        (int(fold), split, encoding, int(n_features), arr.shape[0], arr.shape[1], str(arr.dtype), arr.tobytes())
    )

def _s03_save_target(conn, fold, split, arr):
    conn.execute(
        "INSERT INTO encoded_targets (fold,split,rows,dtype,data) VALUES (?,?,?,?,?)",
        (int(fold), split, len(arr), str(arr.dtype), arr.tobytes())
    )

def _s03_impute_fold(train_df, val_df):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor, XGBClassifier
    train_df, val_df = train_df.copy(), val_df.copy()
    print(f"\n  Imputation (fit on training rows only):")
    for col, ptype in zip(IMPUTE_COLS, IMPUTE_TYPES):
        n_tr = train_df[col].isna().sum(); n_va = val_df[col].isna().sum()
        print(f"    [{col}]  type={ptype}  train_NaN={n_tr:,}  val_NaN={n_va:,}")
        if n_tr == 0 and n_va == 0:
            print(f"      -> No NaN, skipping."); continue
        num_feats = [c for c in train_df.columns
                     if c not in [col, TARGET, TEXT_COL, DX_COL]
                     and pd.api.types.is_numeric_dtype(train_df[c])]
        pre      = Pipeline([('imp', SimpleImputer(strategy='median'))])
        tr_known = train_df[train_df[col].notna()]
        tr_miss  = train_df[train_df[col].isna()]
        va_miss  = val_df[val_df[col].isna()]
        if len(tr_known) == 0:
            print(f"      -> [!] No known train values -- skipping."); continue
        X_tr_k = pre.fit_transform(tr_known[num_feats])
        if ptype == 'classification':
            le    = LabelEncoder()
            y_trk = le.fit_transform(tr_known[col].astype(str))
            mdl   = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbosity=0)
            mdl.fit(X_tr_k, y_trk)
            def predict_col(X, _le=le, _mdl=mdl):
                return _le.inverse_transform(np.round(_mdl.predict(X)).astype(int))
        else:
            mdl = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
            mdl.fit(X_tr_k, tr_known[col].values)
            predict_col = mdl.predict
        if len(tr_miss) > 0:
            train_df.loc[train_df[col].isna(), col] = predict_col(pre.transform(tr_miss[num_feats]))
            print(f"      -> Filled {len(tr_miss):,} train NaN")
        if len(va_miss) > 0:
            val_df.loc[val_df[col].isna(), col] = predict_col(pre.transform(va_miss[num_feats]))
            print(f"      -> Filled {len(va_miss):,} val NaN")
    return train_df, val_df

def _s03_onehot_fold(train_df, val_df, col, fold):
    train_df, val_df = train_df.copy(), val_df.copy()
    cats = sorted(train_df[col].dropna().unique())
    print(f"    [{col}]  fold {fold} train categories: {cats}")
    for cat in cats:
        cname = f'{col}__{cat}'
        train_df[cname] = (train_df[col] == cat).astype(int)
        val_df[cname]   = (val_df[col]   == cat).astype(int)
    train_df.drop(columns=[col], inplace=True)
    val_df.drop(columns=[col], inplace=True)
    return train_df, val_df

def _s03_onehot_all(train_df, val_df, fold):
    print(f"\n  Fold-wise one-hot (train categories only -- no leakage):")
    for col in ['anesthetic_type'] + CATEGORICAL_FOLD_COLS:
        train_df, val_df = _s03_onehot_fold(train_df, val_df, col, fold)
    return train_df, val_df

def _s03_icd10_chapter(code):
    """Map an ICD-10 code string to its chapter integer (0 = unknown/NaN).
    Handles formats like '(m199) description' or 'M19.9' by finding the
    first alphabetic character in the string.
    """
    if not isinstance(code, str) or len(code) == 0:
        return 0
    # Find first alphabetic character (handles '(m199)...' format after normalization)
    letters = [ch for ch in code.upper() if ch.isalpha()]
    if not letters:
        return 0
    c = letters[0]
    mapping = {
        'A': 1, 'B': 1,
        'C': 2, 'D': 2,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 9,
        'J': 10,
        'K': 11,
        'L': 12,
        'M': 13,
        'N': 14,
        'O': 15,
        'P': 16,
        'Q': 17,
        'R': 18,
        'S': 19, 'T': 19,
        'V': 20, 'W': 20, 'X': 20, 'Y': 20,
        'Z': 21,
        'U': 22,
    }
    return mapping.get(c, 0)

def _s03_encode_icd10_onehot(train_dx, val_dx):
    """
    Map most_responsible_dx to ICD-10 chapter integers then one-hot encode.
    Train-fold categories only -- unseen chapters in val become all-zeros rows.
    Returns (X_tr_icd, X_va_icd) as float32 arrays.
    """
    tr_chapters = np.array([_s03_icd10_chapter(x) for x in train_dx])
    va_chapters = np.array([_s03_icd10_chapter(x) for x in val_dx])
    train_cats  = sorted(set(tr_chapters))
    cat_to_idx  = {cat: i for i, cat in enumerate(train_cats)}
    n_cats = len(train_cats)

    def _to_onehot(chapters):
        mat = np.zeros((len(chapters), n_cats), dtype=np.float32)
        for i, ch in enumerate(chapters):
            idx = cat_to_idx.get(ch, None)
            if idx is not None:
                mat[i, idx] = 1.0
        return mat

    return _to_onehot(tr_chapters), _to_onehot(va_chapters)

def _s03_encode_bert_pca(X_struct_tr, X_struct_va, emb_tr, emb_va, n):
    """PCA fit on train fold only -- no leakage into val."""
    from sklearn.decomposition import PCA
    n_comp = min(n, emb_tr.shape[1], emb_tr.shape[0])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_bert_tr = pca.fit_transform(emb_tr)
    X_bert_va = pca.transform(emb_va)
    var_exp   = pca.explained_variance_ratio_.sum() * 100
    X_tr = np.hstack([X_struct_tr, X_bert_tr]).astype(np.float32)
    X_va = np.hstack([X_struct_va, X_bert_va]).astype(np.float32)
    return X_tr, X_va, n_comp, var_exp

def _s03_rag_augment(X_tr_bert, X_va_bert, y_tr, svc_tr, svc_va):
    """
    Build FAISS index on train BERT embeddings; retrieve k=RAG_K neighbours.
    Returns:
        X_tr_rag : hstack([X_tr_bert, rag_nn_tr, rag_svc_tr])
        X_va_rag : hstack([X_va_bert, rag_nn_va, rag_svc_va])
    """
    import faiss

    d = X_tr_bert.shape[1]

    # L2-normalise copies (cosine sim via inner product on unit vectors)
    emb_tr = X_tr_bert.copy().astype(np.float32)
    emb_va = X_va_bert.copy().astype(np.float32)

    norms_tr = np.linalg.norm(emb_tr, axis=1, keepdims=True)
    norms_tr[norms_tr == 0] = 1.0
    emb_tr /= norms_tr

    norms_va = np.linalg.norm(emb_va, axis=1, keepdims=True)
    norms_va[norms_va == 0] = 1.0
    emb_va /= norms_va

    # Build FAISS index
    index = faiss.IndexFlatIP(d)
    index.add(emb_tr)

    # Train: search k+1 (self included); use I_tr[:,1:k+1] to exclude self
    k = RAG_K
    _, I_tr = index.search(emb_tr, k + 1)
    # Val: search k
    _, I_va = index.search(emb_va, k)

    # Nearest-neighbour deviation rates
    y_tr_arr = np.array(y_tr, dtype=np.float32)
    rag_nn_tr = np.array([y_tr_arr[I_tr[i, 1:k+1]].mean() for i in range(len(emb_tr))], dtype=np.float32)
    rag_nn_va = np.array([y_tr_arr[I_va[i, :k]].mean()   for i in range(len(emb_va))], dtype=np.float32)

    # Service-level deviation rates (fit on train, apply to val)
    global_rate = float(y_tr_arr.mean())
    svc_tr_arr  = np.array(svc_tr)
    svc_va_arr  = np.array(svc_va)
    service_rate_dict = {}
    for svc in np.unique(svc_tr_arr):
        mask = svc_tr_arr == svc
        service_rate_dict[svc] = float(y_tr_arr[mask].mean())

    rag_svc_tr = np.array([service_rate_dict.get(s, global_rate) for s in svc_tr_arr], dtype=np.float32)
    rag_svc_va = np.array([service_rate_dict.get(s, global_rate) for s in svc_va_arr], dtype=np.float32)

    # Concatenate RAG features
    X_tr_rag = np.hstack([X_tr_bert, rag_nn_tr.reshape(-1, 1), rag_svc_tr.reshape(-1, 1)]).astype(np.float32)
    X_va_rag = np.hstack([X_va_bert, rag_nn_va.reshape(-1, 1), rag_svc_va.reshape(-1, 1)]).astype(np.float32)

    return X_tr_rag, X_va_rag

def run_stage03():
    if _s03_is_done():
        print(f"  [>>]  Stage 03 already complete ({ENCODED_DB} has all expected matrices). Skipping.")
        with sqlite3.connect(ENCODED_DB) as conn:
            for row in conn.execute("SELECT fold,split,encoding,n_features,rows,cols FROM encoded_matrices ORDER BY fold,split,encoding,n_features"):
                print(f"    fold={row[0]} split={row[1]} encoding={row[2]:<25} n={row[3]:>3}  shape=({row[4]},{row[5]})")
        return

    tee = _Tee(f'{LOG_DIR}/task3_03_fold_encoding.log')
    sys.stdout = tee
    try:
        sep("STAGE 03 -- FOLD ENCODING  (Task 3)")
        print(f"  FEATURES_PER_COL = {FEATURES_PER_COL}")
        print(f"  RAG_K            = {RAG_K}")
        print(f"  ICD-10 chapter one-hot encoding (fold-wise, train categories only)")
        print(f"  BERT PCA (train-only, no leakage) stored per fold -- no PCA recomputation in Stage 04.")

        if not _s01_is_done():
            print(f"  [X] Stage 01 not complete -- run Stage 01 first."); return

        # -- Load BERT caches (optional -- skipped gracefully if missing) -------
        bert_cache_s03 = {}
        sep("BERT CACHE CHECK")
        for tid, (method, fname) in S02_TASKS.items():
            path = os.path.join(BERT_DIR, fname)
            if os.path.exists(path):
                bert_cache_s03[method] = np.load(path)
                print(f"  [OK] {fname}  shape={bert_cache_s03[method].shape}")
            else:
                print(f"  [!] {fname} not found -- BERT encodings skipped (run Stage 02 first)")
        if not bert_cache_s03:
            print(f"  [!] No BERT caches available -- only struct_only will be stored.")

        sep("LOAD CLEAN TABLE")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {CLEAN_TABLE}", conn)
        df = df[df[TARGET].notna()].copy().reset_index(drop=True)
        dropped_leak = [c for c in EXCLUDE_COLS if c in df.columns]
        df.drop(columns=dropped_leak, inplace=True)
        print(f"  Loaded   : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Excluded : {dropped_leak}  (intraoperative leakage)")
        print(f"  NaN (fold-imputed)    : { {col: int(df[col].isna().sum()) for col in IMPUTE_COLS if col in df.columns} }")
        print(f"  Categorical (fold OHE): {['anesthetic_type'] + CATEGORICAL_FOLD_COLS}")
        print(f"  Text column           : {TEXT_COL}")
        print(f"  ICD-10 column         : {DX_COL}")

        sep("LOAD FOLD INDICES")
        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {FOLD_TABLE}", conn)
        # Folds 0-4 (KFold) + fold 5 (temporal)
        all_folds = sorted(fold_df['fold'].unique())
        fold_indices = {}
        for fold in all_folds:
            tr_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values
            va_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values
            fold_indices[fold] = (tr_idx, va_idx)
            note = '  (temporal)' if fold == 5 else ''
            print(f"  Fold {fold}: train={len(tr_idx):,}  val={len(va_idx):,}{note}")

        if os.path.exists(ENCODED_DB):
            os.remove(ENCODED_DB); print(f"\n  Removed existing {ENCODED_DB} -- starting fresh.")
        with sqlite3.connect(ENCODED_DB) as conn:
            _s03_init_db(conn)

        sep("ENCODING LOOP -- fold -> impute -> one-hot -> ICD-10 -> struct matrix -> BERT PCA + RAG -> save")
        for fold, (tr_idx, va_idx) in fold_indices.items():
            fold_label = f"FOLD {fold}  train={len(tr_idx):,}  val={len(va_idx):,}" + (' (temporal)' if fold == 5 else '')
            sep(fold_label, char='-')
            train_base = df.iloc[tr_idx].copy().reset_index(drop=True)
            val_base   = df.iloc[va_idx].copy().reset_index(drop=True)

            # -- Fold-wise imputation (age, BMI, anesthetic_type) --------------
            train_base, val_base = _s03_impute_fold(train_base, val_base)

            # -- Save case_service series BEFORE one-hot (needed for RAG) ------
            case_service_train = train_base['case_service'].copy()
            case_service_val   = val_base['case_service'].copy()

            # -- Save y_tr for RAG deviation rates ----------------------------
            y_tr_labels = train_base[TARGET].values.astype(np.float32)

            # -- Fold-wise one-hot: anesthetic_type, case_service, surgical_location
            train_base, val_base = _s03_onehot_all(train_base, val_base, fold)

            # -- ICD-10 chapter one-hot encoding -------------------------------
            print(f"\n  ICD-10 chapter one-hot encoding (fold-wise):")
            train_dx = train_base[DX_COL].astype(str).tolist()
            val_dx   = val_base[DX_COL].astype(str).tolist()
            X_icd_tr, X_icd_va = _s03_encode_icd10_onehot(train_dx, val_dx)
            print(f"    ICD-10 one-hot: train={X_icd_tr.shape}  val={X_icd_va.shape}")

            # -- Targets -------------------------------------------------------
            y_train = train_base[TARGET].values.astype(np.float32)
            y_val   = val_base[TARGET].values.astype(np.float32)
            n_dev_tr = y_train.sum(); n_dev_va = y_val.sum()
            print(f"\n  y_train: n={len(y_train):,}  pos={n_dev_tr:.0f}  rate={n_dev_tr/len(y_train)*100:.2f}%")
            print(f"  y_val  : n={len(y_val):,}  pos={n_dev_va:.0f}  rate={n_dev_va/len(y_val)*100:.2f}%")
            with sqlite3.connect(ENCODED_DB) as conn:
                _s03_save_target(conn, fold, 'train', y_train)
                _s03_save_target(conn, fold, 'val',   y_val)
                conn.commit()

            # -- Build struct matrix: one-hot columns + ICD-10 (no TEXT_COL, DX_COL, TARGET, case_id)
            struct_cols = [c for c in train_base.columns if c not in [TEXT_COL, DX_COL, TARGET, 'case_id']]
            X_tr_struct = np.hstack([
                train_base[struct_cols].values.astype(np.float32),
                X_icd_tr
            ])
            X_va_struct = np.hstack([
                val_base[struct_cols].values.astype(np.float32),
                X_icd_va
            ])
            print(f"\n  struct_only (with ICD-10 OHE): train={X_tr_struct.shape}  val={X_va_struct.shape}")

            with sqlite3.connect(ENCODED_DB) as conn:
                _s03_save_matrix(conn, fold, 'train', 'struct_only', 0, X_tr_struct)
                _s03_save_matrix(conn, fold, 'val',   'struct_only', 0, X_va_struct)
                conn.commit()

            # -- BERT PCA + RAG encodings (per method, per n) -----------------
            if bert_cache_s03:
                print(f"\n  BERT PCA + RAG encodings (PCA fit on train fold only -- no leakage):")
            for method, emb in bert_cache_s03.items():
                emb_tr_full = emb[tr_idx].astype(np.float32)
                emb_va_full = emb[va_idx].astype(np.float32)
                for n in FEATURES_PER_COL:
                    # PCA BERT features + struct
                    X_tr, X_va, n_comp, var_exp = _s03_encode_bert_pca(
                        X_tr_struct, X_va_struct, emb_tr_full, emb_va_full, n
                    )
                    print(f"    n={n:>3}  {method:<16}  PCA({n_comp}) on {emb_tr_full.shape[1]}-d -> {var_exp:.1f}% var  train={X_tr.shape}  val={X_va.shape}")
                    with sqlite3.connect(ENCODED_DB) as conn:
                        _s03_save_matrix(conn, fold, 'train', method, n, X_tr)
                        _s03_save_matrix(conn, fold, 'val',   method, n, X_va)
                        conn.commit()

                    # RAG augmentation using just the BERT PCA part
                    # Extract the BERT-PCA portion (last n_comp columns of X_tr/X_va)
                    n_struct = X_tr_struct.shape[1]
                    X_tr_bert_part = X_tr[:, n_struct:n_struct + n_comp]
                    X_va_bert_part = X_va[:, n_struct:n_struct + n_comp]

                    try:
                        X_tr_rag, X_va_rag = _s03_rag_augment(
                            X_tr_bert_part, X_va_bert_part,
                            y_tr_labels,
                            case_service_train.values, case_service_val.values
                        )
                        # Prepend struct features: [struct | bert_pca | rag_nn | rag_svc]
                        X_tr_rag_full = np.hstack([X_tr_struct, X_tr_rag]).astype(np.float32)
                        X_va_rag_full = np.hstack([X_va_struct, X_va_rag]).astype(np.float32)
                        rag_enc = f'rag_{method}'
                        print(f"    n={n:>3}  {rag_enc:<16}  RAG(k={RAG_K})  train={X_tr_rag_full.shape}  val={X_va_rag_full.shape}")
                        with sqlite3.connect(ENCODED_DB) as conn:
                            _s03_save_matrix(conn, fold, 'train', rag_enc, n, X_tr_rag_full)
                            _s03_save_matrix(conn, fold, 'val',   rag_enc, n, X_va_rag_full)
                            conn.commit()
                    except Exception as e:
                        print(f"    [!] RAG augmentation failed for {method} n={n}: {e}")

        sep("SUMMARY")
        with sqlite3.connect(ENCODED_DB) as conn:
            n_mat = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            n_tgt = conn.execute("SELECT COUNT(*) FROM encoded_targets").fetchone()[0]
            print(f"  encoded_matrices : {n_mat} rows")
            print(f"  encoded_targets  : {n_tgt} rows")
            print(f"\n  {'Fold':<6} {'Split':<8} {'Encoding':<25} {'n_features':>10} {'Rows':>8} {'Cols':>6}")
            print(f"  {'-'*67}")
            for row in conn.execute("SELECT fold,split,encoding,n_features,rows,cols FROM encoded_matrices ORDER BY fold,split,encoding,n_features"):
                print(f"  {row[0]:<6} {row[1]:<8} {row[2]:<25} {row[3]:>10} {row[4]:>8,} {row[5]:>6}")
        print(f"  DB size: {os.path.getsize(ENCODED_DB)/1024/1024:.1f} MB")
        print(f"  [OK] Stage 03 complete.  Log -> {LOG_DIR}/task3_03_fold_encoding.log")

    finally:
        tee.close()


# =============================================================================
# STAGE 04 -- MODELING  (binary classification, all models run automatically)
# =============================================================================

def run_stage04():

    import optuna
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  f1_score, precision_score, recall_score,
                                  brier_score_loss)
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.base import BaseEstimator
    import xgboost as xgb
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.mixed_precision import set_global_policy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # -- TF GPU setup ----------------------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            set_global_policy('mixed_float16')
            print(f"  [OK] TF GPU: {[g.name for g in gpus]}  mixed_float16 enabled")
        except RuntimeError as e:
            print(f"  [!] TF GPU config error: {e}")
    else:
        print("  [!] No TF GPU detected -- MLP will run on CPU")

    # -- XGBoost CUDA runtime check --------------------------------------------
    xgb_device = 'cpu'
    try:
        _probe = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1, verbosity=0)
        _probe.fit(np.zeros((10, 2)), np.zeros(10, dtype=int))
        xgb_device = 'cuda'
        print(f"  [OK] XGBoost CUDA available")
    except Exception:
        print(f"  [!] XGBoost CUDA unavailable -- falling back to CPU")

    tee = _Tee(f'{LOG_DIR}/task3_04_modeling.log')
    sys.stdout = tee
    try:
        sep("STAGE 04 -- MODELING  (Task 3: Procedure Deviation Detection)")
        sep("CONFIG SNAPSHOT")
        print(f"  FEATURES_PER_COL   = {FEATURES_PER_COL}  (all values in DB -- no PCA in Stage 04)")
        print(f"  N_SPLITS           = {N_SPLITS}  + temporal fold 5  /  RANDOM_STATE = {RANDOM_STATE}")
        print(f"  N_TRIALS           = {N_TRIALS}  /  TUNE_SET_FRACTION = {TUNE_SET_FRACTION}")
        print(f"  CLINICAL_THRESHOLD = {CLINICAL_THRESHOLD}")
        print(f"  RAG_K              = {RAG_K}")
        print(f"  Models (all run automatically): {ALL_MODELS}")
        print(f"  MLP: AdamW + BatchNorm + variable depth ({OPTUNA_MLP_N_LAYERS_LOW}-{OPTUNA_MLP_N_LAYERS_HIGH} layers, Optuna-tuned)")
        print(f"  MLP training: epochs={MLP_EPOCHS_FINAL}/{MLP_EPOCHS_OPTUNA}  batch={MLP_BATCH_SIZE}  clipnorm={MLP_CLIPNORM}")
        print(f"  XGBoost: device={xgb_device}  early_stopping_rounds={XGB_EARLY_STOPPING_ROUNDS}")
        print(f"\n  Per-model artifacts: {LOG_DIR}/task3_<model>.log  and  {LOG_DIR}/task3_<model>.pdf")

        # -- Verify prerequisites ----------------------------------------------
        sep("VERIFY PREREQUISITES")
        prereqs_ok = True
        if not _s01_is_done():
            print(f"  [X] Stage 01 not complete."); prereqs_ok = False
        else:
            print(f"  [OK] {DB_PATH}")
        if not _s03_is_done():
            print(f"  [X] Stage 03 not complete."); prereqs_ok = False
        else:
            print(f"  [OK] {ENCODED_DB}")
        if not prereqs_ok: return

        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {FOLD_TABLE}", conn)
        all_folds = sorted(fold_df['fold'].unique())
        fold_indices = {}
        for fold in all_folds:
            tr_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values
            va_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values
            fold_indices[fold] = (tr_idx, va_idx)
        print(f"  [OK] Fold indices loaded ({len(all_folds)} folds: {all_folds})")

        with sqlite3.connect(ENCODED_DB) as conn:
            db_combos = set(
                (row[0], row[1]) for row in
                conn.execute("SELECT DISTINCT encoding, n_features FROM encoded_matrices").fetchall()
            )
        print(f"  [OK] {len(db_combos)} (encoding, n) combos found in {ENCODED_DB}")

        # -- Run ALL models automatically (no interactive prompt) --------------
        MODEL_LIST = ALL_MODELS[:]
        print(f"\n  Running all models: {MODEL_LIST}")

        # -- Inner helpers -----------------------------------------------------

        def load_fold_matrices(fold):
            """Load all encoded matrices for a given fold in a single DB connection."""
            result = {}
            with sqlite3.connect(ENCODED_DB) as conn:
                rows = conn.execute(
                    "SELECT split, encoding, n_features, rows, cols, dtype, data FROM encoded_matrices WHERE fold=?",
                    (int(fold),)   # cast to int: np.int64 does not bind to SQLite integer
                ).fetchall()
            for split, encoding, n_features, n_rows, n_cols, dtype, data in rows:
                result[(split, encoding, int(n_features))] = (
                    np.frombuffer(data, dtype=dtype).reshape(n_rows, n_cols).copy()
                )
            return result

        def load_target(fold, split):
            with sqlite3.connect(ENCODED_DB) as conn:
                row = conn.execute(
                    "SELECT rows,dtype,data FROM encoded_targets WHERE fold=? AND split=?",
                    (int(fold), split)   # cast to int: np.int64 does not bind to SQLite integer
                ).fetchone()
            if row is None: raise ValueError(f"Target not found: fold={fold} split={split}")
            return np.frombuffer(row[2], dtype=row[1]).copy()

        def save_db(df_save, table):
            with sqlite3.connect(RESULT_DB, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                df_save.to_sql(table, conn, if_exists='append', index=False)

        def _s04_done_combos():
            """Return the set of (fold, encoding, n_features, model) already saved."""
            if not os.path.exists(RESULT_DB):
                return set()
            try:
                with sqlite3.connect(RESULT_DB, timeout=30) as conn:
                    rows = conn.execute(
                        "SELECT fold, encoding, n_features, model FROM metrics"
                    ).fetchall()
                return {(int(r[0]), r[1], int(r[2]), r[3]) for r in rows}
            except Exception:
                return set()

        done_combos  = _s04_done_combos()
        resume_mode  = True   # always resume; skip completed combos automatically
        if done_combos:
            print(f"\n  [>>]  Resume mode: {len(done_combos)} already-completed combo(s) found -- will be skipped.")
        else:
            print(f"  [OK]  No prior results found -- running all combos fresh.")

        def delete_existing(fold, encoding, n_features, model_name):
            if not os.path.exists(RESULT_DB): return
            with sqlite3.connect(RESULT_DB, timeout=30) as conn:
                for table in ['metrics', 'predictions', 'feature_importance', 'hyperparameter']:
                    try:
                        conn.execute(
                            f"DELETE FROM {table} WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                            (int(fold), encoding, int(n_features), model_name)
                        )
                    except Exception: pass
                conn.commit()

        def compute_metrics(y_true, y_prob, threshold=CLINICAL_THRESHOLD):
            """AUC-ROC, AUC-PR, F1/Precision/Recall at threshold, Brier, clinical counts."""
            y_true = np.array(y_true, dtype=int)
            y_prob = np.array(y_prob, dtype=float)

            # Guard against degenerate cases (all same label)
            if len(np.unique(y_true)) < 2:
                auc_roc = 0.5
                auc_pr  = float(y_true.mean())
            else:
                auc_roc = float(roc_auc_score(y_true, y_prob))
                auc_pr  = float(average_precision_score(y_true, y_prob))

            y_pred  = (y_prob >= threshold).astype(int)
            f1      = float(f1_score(y_true, y_pred, zero_division=0))
            prec    = float(precision_score(y_true, y_pred, zero_division=0))
            rec     = float(recall_score(y_true, y_pred, zero_division=0))
            brier   = float(brier_score_loss(y_true, y_prob))

            n_flagged            = int(y_pred.sum())
            n_caught             = int((y_pred & y_true).sum())
            n_total_deviations   = int(y_true.sum())

            return {
                'auc_roc':             auc_roc,
                'auc_pr':              auc_pr,
                'f1':                  f1,
                'precision':           prec,
                'recall':              rec,
                'brier':               brier,
                'n_flagged':           n_flagged,
                'n_caught':            n_caught,
                'n_total_deviations':  n_total_deviations,
                'threshold':           threshold,
            }

        def build_model(name, params=None):
            p = params or {}
            RS = RANDOM_STATE
            if name == 'ridge':
                return LogisticRegression(
                    C=p.get('C', 1.0), penalty='l2', solver='lbfgs',
                    max_iter=200, tol=1e-2, class_weight='balanced',
                    random_state=RS
                )
            if name == 'randomforest':
                return RandomForestClassifier(
                    n_estimators=p.get('n_estimators', 200),
                    max_depth=p.get('max_depth', 6),
                    max_features=p.get('max_features', 'sqrt'),
                    min_samples_split=p.get('min_samples_split', 2),
                    min_samples_leaf=p.get('min_samples_leaf', 1),
                    class_weight='balanced',
                    random_state=RS, n_jobs=-1
                )
            if name == 'xgboost':
                return xgb.XGBClassifier(
                    n_estimators=p.get('n_estimators', 200),
                    learning_rate=p.get('learning_rate', 0.05),
                    max_depth=p.get('max_depth', 4),
                    subsample=p.get('subsample', 0.8),
                    colsample_bytree=p.get('colsample_bytree', 0.8),
                    reg_alpha=p.get('reg_alpha', 0.01),
                    reg_lambda=p.get('reg_lambda', 1.0),
                    scale_pos_weight=p.get('scale_pos_weight', 13.7),
                    tree_method=XGB_TREE_METHOD,
                    device=xgb_device,
                    eval_metric='aucpr',
                    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                    random_state=RS, n_jobs=-1
                )
            if name == 'lightgbm':
                return lgb.LGBMClassifier(
                    n_estimators=p.get('n_estimators', 200),
                    learning_rate=p.get('learning_rate', 0.05),
                    max_depth=p.get('max_depth', -1),
                    num_leaves=p.get('num_leaves', 31),
                    subsample=p.get('subsample', 0.8),
                    colsample_bytree=p.get('colsample_bytree', 0.8),
                    reg_alpha=p.get('reg_alpha', 0.01),
                    reg_lambda=p.get('reg_lambda', 1.0),
                    is_unbalance=True,
                    verbose=-1,
                    random_state=RS, n_jobs=-1
                )
            if name == 'mlp':
                act      = p.get('activation', 'relu')
                lr       = p.get('lr', 1e-3)
                wd       = p.get('weight_decay', 1e-4)
                n_layers = p.get('n_layers', 2)
                opt = AdamW(learning_rate=lr, weight_decay=wd,
                            clipnorm=MLP_CLIPNORM if MLP_CLIPNORM is not None else None)
                layer_sizes = [OPTUNA_MLP_UNITS1_HIGH, OPTUNA_MLP_UNITS2_HIGH, OPTUNA_MLP_UNITS3_HIGH]
                layer_list  = [Input(shape=(p.get('input_dim', 64),))]
                for i in range(n_layers):
                    units   = p.get(f'units{i+1}', layer_sizes[i] // 2)
                    dropout = p.get(f'dropout{i+1}', 0.1)
                    layer_list.append(Dense(units, activation=act))
                    layer_list.append(BatchNormalization())
                    layer_list.append(Dropout(dropout))
                layer_list.append(Dense(1, activation='sigmoid', dtype='float32'))
                mdl = Sequential(layer_list)
                mdl.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=[tf.keras.metrics.AUC(name='auc')])
                return mdl
            raise ValueError(f"Unknown model: {name}")

        def fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=None,
                        final=False, scale_pos_weight=1.0):
            """Fit and predict -- returns (y_prob, train_time_s, infer_time_s)."""
            t_train_start = time.perf_counter()

            if name == 'mlp':
                Xtr_f = Xtr_sc.astype(np.float32); ytr_f = ytr.astype(np.float32)
                # Compute sample weights for class imbalance
                n_neg  = (ytr_f == 0).sum(); n_pos = (ytr_f == 1).sum()
                w_pos  = (n_neg / max(n_pos, 1))
                sw     = np.where(ytr_f == 1, w_pos, 1.0).astype(np.float32)

                if final and yva is not None:
                    Xva_f = Xva_sc.astype(np.float32); yva_f = yva.astype(np.float32)
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f, sw))
                                .shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE)
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f))
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    cbs = [
                        EarlyStopping(monitor='val_auc', patience=MLP_PATIENCE_ES,
                                      restore_best_weights=True, verbose=0, mode='max'),
                        ReduceLROnPlateau(monitor='val_auc', factor=MLP_LR_DECAY_FACTOR,
                                         patience=MLP_PATIENCE_LR, min_lr=MLP_MIN_LR,
                                         verbose=0, mode='max'),
                    ]
                    mdl.fit(train_ds, validation_data=val_ds,
                            epochs=MLP_EPOCHS_FINAL, callbacks=cbs, verbose=0)
                    train_time_s = time.perf_counter() - t_train_start
                    t_infer_start = time.perf_counter()
                    y_prob = mdl.predict(val_ds, verbose=0).flatten()
                    infer_time_s = time.perf_counter() - t_infer_start
                else:
                    mdl.fit(Xtr_f, ytr_f, sample_weight=sw,
                            epochs=MLP_EPOCHS_OPTUNA, batch_size=MLP_BATCH_SIZE, verbose=0)
                    train_time_s = time.perf_counter() - t_train_start
                    t_infer_start = time.perf_counter()
                    y_prob = mdl.predict(Xva_sc.astype(np.float32), verbose=0).flatten()
                    infer_time_s = time.perf_counter() - t_infer_start

            elif name == 'xgboost':
                if yva is not None:
                    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                else:
                    mdl.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                y_prob = mdl.predict_proba(Xva)[:, 1]
                infer_time_s = time.perf_counter() - t_infer_start

            elif name == 'lightgbm':
                if yva is not None:
                    mdl.fit(Xtr, ytr,
                            eval_set=[(Xva, yva)],
                            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False),
                                       lgb.log_evaluation(-1)])
                else:
                    mdl.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                y_prob = mdl.predict_proba(Xva)[:, 1]
                infer_time_s = time.perf_counter() - t_infer_start

            elif name == 'randomforest':
                mdl.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                y_prob = mdl.predict_proba(Xva)[:, 1]
                infer_time_s = time.perf_counter() - t_infer_start

            else:  # ridge, lasso, elasticnet (LogisticRegression)
                mdl.fit(Xtr_sc, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                y_prob = mdl.predict_proba(Xva_sc)[:, 1]
                infer_time_s = time.perf_counter() - t_infer_start

            return y_prob, train_time_s, infer_time_s

        # -- Optuna pruning callback for MLP -----------------------------------
        class _OptunaPruningCallback(tf.keras.callbacks.Callback):
            def __init__(self, trial):
                super().__init__()
                self._trial = trial
            def on_epoch_end(self, epoch, logs=None):
                val_auc = (logs or {}).get('val_auc')
                if val_auc is None:
                    return
                # Negate because we minimise (maximise AUC -> minimise -AUC)
                self._trial.report(-float(val_auc), epoch)
                if self._trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        def make_objective(name, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva):
            def obj(trial):
                n_neg  = (ytr == 0).sum(); n_pos = (ytr == 1).sum()
                pos_wt = float(n_neg / max(n_pos, 1))

                if name == 'ridge':
                    p = {'C': trial.suggest_float('C', OPTUNA_C_LOW, OPTUNA_C_HIGH, log=True)}
                elif name == 'elasticnet':
                    p = {
                        'C':        trial.suggest_float('C',        OPTUNA_EN_C_LOW,        OPTUNA_EN_C_HIGH,        log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', OPTUNA_EN_L1_RATIO_LOW, OPTUNA_EN_L1_RATIO_HIGH),
                    }
                elif name == 'lightgbm':
                    p = {
                        'n_estimators':     trial.suggest_int(  'n_estimators',     OPTUNA_LGB_N_EST_LOW,       OPTUNA_LGB_N_EST_HIGH),
                        'learning_rate':    trial.suggest_float('learning_rate',    OPTUNA_LGB_LR_LOW,          OPTUNA_LGB_LR_HIGH,          log=True),
                        'max_depth':        trial.suggest_int(  'max_depth',        OPTUNA_LGB_MAX_DEPTH_LOW,   OPTUNA_LGB_MAX_DEPTH_HIGH),
                        'num_leaves':       trial.suggest_int(  'num_leaves',       OPTUNA_LGB_NUM_LEAVES_LOW,  OPTUNA_LGB_NUM_LEAVES_HIGH),
                        'subsample':        trial.suggest_float('subsample',        OPTUNA_LGB_SUBSAMPLE_LOW,   OPTUNA_LGB_SUBSAMPLE_HIGH),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', OPTUNA_LGB_COLSAMPLE_LOW,   OPTUNA_LGB_COLSAMPLE_HIGH),
                        'reg_alpha':        trial.suggest_float('reg_alpha',        OPTUNA_LGB_REG_ALPHA_LOW,   OPTUNA_LGB_REG_ALPHA_HIGH,   log=True),
                        'reg_lambda':       trial.suggest_float('reg_lambda',       OPTUNA_LGB_REG_LAMBDA_LOW,  OPTUNA_LGB_REG_LAMBDA_HIGH,  log=True),
                    }
                elif name == 'randomforest':
                    p = {
                        'n_estimators':      trial.suggest_int('n_estimators', OPTUNA_RF_N_EST_LOW, OPTUNA_RF_N_EST_HIGH),
                        'max_depth':         trial.suggest_int('max_depth', OPTUNA_RF_MAX_DEPTH_LOW, OPTUNA_RF_MAX_DEPTH_HIGH),
                        'max_features':      trial.suggest_categorical('max_features', OPTUNA_RF_MAX_FEATURES),
                        'min_samples_split': trial.suggest_int('min_samples_split', OPTUNA_RF_MIN_SAMPLES_SPLIT_LOW, OPTUNA_RF_MIN_SAMPLES_SPLIT_HIGH),
                        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', OPTUNA_RF_MIN_SAMPLES_LEAF_LOW, OPTUNA_RF_MIN_SAMPLES_LEAF_HIGH),
                    }
                elif name == 'xgboost':
                    p = {
                        'n_estimators':     trial.suggest_int(  'n_estimators',     OPTUNA_XGB_N_EST_LOW,       OPTUNA_XGB_N_EST_HIGH),
                        'learning_rate':    trial.suggest_float('learning_rate',     OPTUNA_XGB_LR_LOW,          OPTUNA_XGB_LR_HIGH,          log=True),
                        'max_depth':        trial.suggest_int(  'max_depth',         OPTUNA_XGB_MAX_DEPTH_LOW,   OPTUNA_XGB_MAX_DEPTH_HIGH),
                        'subsample':        trial.suggest_float('subsample',         OPTUNA_XGB_SUBSAMPLE_LOW,   OPTUNA_XGB_SUBSAMPLE_HIGH),
                        'colsample_bytree': trial.suggest_float('colsample_bytree',  OPTUNA_XGB_COLSAMPLE_LOW,   OPTUNA_XGB_COLSAMPLE_HIGH),
                        'reg_alpha':        trial.suggest_float('reg_alpha',         OPTUNA_XGB_REG_ALPHA_LOW,   OPTUNA_XGB_REG_ALPHA_HIGH,   log=True),
                        'reg_lambda':       trial.suggest_float('reg_lambda',        OPTUNA_XGB_REG_LAMBDA_LOW,  OPTUNA_XGB_REG_LAMBDA_HIGH,  log=True),
                        'scale_pos_weight': pos_wt,
                        'tree_method': XGB_TREE_METHOD, 'device': xgb_device,
                        'early_stopping_rounds': XGB_EARLY_STOPPING_ROUNDS,
                    }
                elif name == 'mlp':
                    tf.keras.backend.clear_session()
                    n_layers = trial.suggest_int('n_layers', OPTUNA_MLP_N_LAYERS_LOW, OPTUNA_MLP_N_LAYERS_HIGH)
                    units_bounds = [
                        (OPTUNA_MLP_UNITS1_LOW, OPTUNA_MLP_UNITS1_HIGH),
                        (OPTUNA_MLP_UNITS2_LOW, OPTUNA_MLP_UNITS2_HIGH),
                        (OPTUNA_MLP_UNITS3_LOW, OPTUNA_MLP_UNITS3_HIGH),
                    ]
                    p = {
                        'input_dim':    Xtr_sc.shape[1],
                        'n_layers':     n_layers,
                        'activation':   trial.suggest_categorical('activation', OPTUNA_MLP_ACTIVATIONS),
                        'lr':           trial.suggest_float('lr', OPTUNA_MLP_LR_LOW, OPTUNA_MLP_LR_HIGH, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', OPTUNA_MLP_WEIGHT_DECAY_LOW, OPTUNA_MLP_WEIGHT_DECAY_HIGH, log=True),
                    }
                    for i in range(OPTUNA_MLP_N_LAYERS_HIGH):
                        lo_u, hi_u = units_bounds[i]
                        p[f'units{i+1}']   = trial.suggest_int(f'units{i+1}', lo_u, hi_u)
                        p[f'dropout{i+1}'] = trial.suggest_float(f'dropout{i+1}', OPTUNA_MLP_DROPOUT_LOW, OPTUNA_MLP_DROPOUT_HIGH)
                else:
                    p = {}

                if name == 'mlp':
                    mdl   = build_model('mlp', p)
                    Xtr_f = Xtr_sc.astype(np.float32)
                    ytr_f = ytr.astype(np.float32)
                    Xva_f = Xva_sc.astype(np.float32)
                    yva_f = yva.astype(np.float32)
                    n_neg  = (ytr_f == 0).sum(); n_pos = (ytr_f == 1).sum()
                    w_pos  = float(n_neg / max(n_pos, 1))
                    sw     = np.where(ytr_f == 1, w_pos, 1.0).astype(np.float32)
                    if MLP_OPTUNA_SUBSET_SIZE and len(Xtr_f) > MLP_OPTUNA_SUBSET_SIZE:
                        rng   = np.random.default_rng(RANDOM_STATE)
                        idx   = rng.choice(len(Xtr_f), MLP_OPTUNA_SUBSET_SIZE, replace=False)
                        Xtr_f = Xtr_f[idx]
                        ytr_f = ytr_f[idx]
                        sw    = sw[idx]
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f, sw))
                                .shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE)
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f))
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    history = mdl.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=MLP_EPOCHS_OPTUNA,
                        callbacks=[_OptunaPruningCallback(trial)],
                        verbose=0,
                    )
                    # Best val_auc seen (negated to fit minimise direction)
                    best_val_auc = max(history.history.get('val_auc', [0.0]))
                    return -float(best_val_auc)
                else:
                    mdl = build_model(name, p)
                    y_prob, _, _ = fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=yva)
                    if len(np.unique(yva)) < 2:
                        return 0.5
                    return -float(roc_auc_score(yva, y_prob))  # negate: Optuna minimises
            return obj

        # -- Per-model PDF artifacts -------------------------------------------
        def save_model_artifacts(model_name, model_rows):
            """Write .log and .pdf for one model, with ROC/PR/calibration/clinical pages."""
            df_m = pd.DataFrame(model_rows)
            log_path = os.path.join(LOG_DIR, f'task3_{model_name}.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n  MODEL: {model_name.upper()}  (Task 3: Procedure Deviation)\n{'='*80}\n\n")
                f.write("PER-FOLD RESULTS\n")
                f.write(f"  {'Encoding':<22} {'n':>6} {'Fold':>5} {'AUC-ROC':>8} {'AUC-PR':>8} "
                        f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'Brier':>7} "
                        f"{'Flagged':>8} {'Caught':>7} {'Train(s)':>10} {'Infer(s)':>10}\n")
                f.write(f"  {'-'*117}\n")
                for _, row in df_m.sort_values(['encoding', 'n_features', 'fold']).iterrows():
                    n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    f.write(f"  {row['encoding']:<22} {n_str:>6} {int(row['fold']):>5} "
                            f"{row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
                            f"{row['f1']:>7.4f} {row['precision']:>7.4f} {row['recall']:>7.4f} "
                            f"{row['brier']:>7.4f} "
                            f"{int(row['n_flagged']):>8,} {int(row['n_caught']):>7,} "
                            f"{row['train_time_s']:>10.2f} {row['infer_time_s']:>10.4f}\n")

                # Aggregated summary
                grp = df_m.groupby(['encoding', 'n_features'])[
                    ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'brier',
                     'train_time_s', 'infer_time_s']
                ].agg(['mean', 'std'])
                grp.columns = ['_'.join(c) for c in grp.columns]
                grp = grp.reset_index()

                f.write(f"\n\nAGGREGATED SUMMARY  (mean +/- std across folds)\n")
                f.write(f"  {'Encoding':<22} {'n':>6} {'AUC-ROC':>9} {'AUC-PR':>9} "
                        f"{'F1':>8} {'Brier':>8} {'Train(s)':>10}\n")
                f.write(f"  {'-'*80}\n")
                for _, row in grp.sort_values('auc_roc_mean', ascending=False).iterrows():
                    n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    f.write(f"  {row['encoding']:<22} {n_str:>6} "
                            f"{row['auc_roc_mean']:>7.4f}+/-{row['auc_roc_std']:.4f} "
                            f"{row['auc_pr_mean']:>7.4f}+/-{row['auc_pr_std']:.4f} "
                            f"{row['f1_mean']:>6.4f}+/-{row['f1_std']:.4f} "
                            f"{row['brier_mean']:>6.4f}+/-{row['brier_std']:.4f} "
                            f"{row['train_time_s_mean']:>10.2f}\n")
            print(f"   task3_{model_name}.log  -> {log_path}")

            pdf_path = os.path.join(LOG_DIR, f'task3_{model_name}.pdf')
            with PdfPages(pdf_path) as pdf:

                # Page 1: ROC curves per fold + mean
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.set_title(f'{model_name.upper()} -- ROC Curves (all folds + mean)', fontweight='bold')
                tprs_all = []
                mean_fpr = np.linspace(0, 1, 100)
                fold_enc_combos = df_m[['encoding', 'n_features']].drop_duplicates().head(6)
                color_cycle = plt.cm.tab10.colors
                for ci, (_, cr) in enumerate(fold_enc_combos.iterrows()):
                    enc_rows = df_m[(df_m['encoding'] == cr['encoding']) & (df_m['n_features'] == cr['n_features'])]
                    color    = color_cycle[ci % 10]
                    enc_label = f"{cr['encoding']} n={int(cr['n_features'])}"
                    aucs = []
                    tprs_enc = []
                    for _, row in enc_rows.iterrows():
                        # Load predictions for this fold/enc/n/model from RESULT_DB
                        try:
                            with sqlite3.connect(RESULT_DB, timeout=30) as conn_r:
                                preds_row = pd.read_sql(
                                    "SELECT y_true, y_prob FROM predictions WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                                    conn_r, params=(int(row['fold']), row['encoding'], int(row['n_features']), model_name)
                                )
                            if len(preds_row) < 2 or len(preds_row['y_true'].unique()) < 2:
                                continue
                            from sklearn.metrics import roc_curve
                            fpr, tpr, _ = roc_curve(preds_row['y_true'], preds_row['y_prob'])
                            auc_val = float(roc_auc_score(preds_row['y_true'], preds_row['y_prob']))
                            ax1.plot(fpr, tpr, alpha=0.3, color=color, linewidth=0.8)
                            tprs_enc.append(np.interp(mean_fpr, fpr, tpr))
                            tprs_enc[-1][0] = 0.0
                            aucs.append(auc_val)
                            tprs_all.extend(tprs_enc[-1:])
                        except Exception:
                            continue
                    if tprs_enc:
                        mean_tpr = np.mean(tprs_enc, axis=0)
                        mean_tpr[-1] = 1.0
                        mean_auc = float(np.mean(aucs))
                        ax1.plot(mean_fpr, mean_tpr, color=color, linewidth=2.0,
                                 label=f'{enc_label}  AUC={mean_auc:.3f}')
                ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
                ax1.set_xlabel('False Positive Rate', fontsize=11)
                ax1.set_ylabel('True Positive Rate', fontsize=11)
                ax1.legend(loc='lower right', fontsize=8)
                ax1.grid(alpha=0.3)
                plt.tight_layout(); pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)

                # Page 2: PR curves per fold + mean
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.set_title(f'{model_name.upper()} -- Precision-Recall Curves (all folds + mean)', fontweight='bold')
                for ci, (_, cr) in enumerate(fold_enc_combos.iterrows()):
                    enc_rows = df_m[(df_m['encoding'] == cr['encoding']) & (df_m['n_features'] == cr['n_features'])]
                    color    = color_cycle[ci % 10]
                    enc_label = f"{cr['encoding']} n={int(cr['n_features'])}"
                    prs_enc   = []
                    aucs_pr   = []
                    mean_rec  = np.linspace(0, 1, 100)
                    for _, row in enc_rows.iterrows():
                        try:
                            with sqlite3.connect(RESULT_DB, timeout=30) as conn_r:
                                preds_row = pd.read_sql(
                                    "SELECT y_true, y_prob FROM predictions WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                                    conn_r, params=(int(row['fold']), row['encoding'], int(row['n_features']), model_name)
                                )
                            if len(preds_row) < 2 or len(preds_row['y_true'].unique()) < 2:
                                continue
                            from sklearn.metrics import precision_recall_curve
                            prec_c, rec_c, _ = precision_recall_curve(preds_row['y_true'], preds_row['y_prob'])
                            ap = float(average_precision_score(preds_row['y_true'], preds_row['y_prob']))
                            ax2.plot(rec_c, prec_c, alpha=0.3, color=color, linewidth=0.8)
                            prec_interp = np.interp(mean_rec, rec_c[::-1], prec_c[::-1])
                            prs_enc.append(prec_interp)
                            aucs_pr.append(ap)
                        except Exception:
                            continue
                    if prs_enc:
                        mean_pr = np.mean(prs_enc, axis=0)
                        mean_ap = float(np.mean(aucs_pr))
                        ax2.plot(mean_rec, mean_pr, color=color, linewidth=2.0,
                                 label=f'{enc_label}  AP={mean_ap:.3f}')
                pos_rate = df_m['n_total_deviations'].sum() / max(df_m['n_flagged'].sum() + (df_m['n_total_deviations'] - df_m['n_caught']).sum(), 1)
                baseline = df_m['n_total_deviations'].iloc[0] / max(len(df_m), 1) if len(df_m) > 0 else 0.068
                ax2.axhline(y=0.068, color='k', linestyle='--', linewidth=0.8, label='Baseline (~6.8%)')
                ax2.set_xlabel('Recall', fontsize=11)
                ax2.set_ylabel('Precision', fontsize=11)
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(alpha=0.3)
                plt.tight_layout(); pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

                # Page 3: Calibration curves (reliability diagram)
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                ax3.set_title(f'{model_name.upper()} -- Calibration Curves (Reliability Diagram)', fontweight='bold')
                ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Perfect calibration')
                for ci, (_, cr) in enumerate(fold_enc_combos.iterrows()):
                    enc_rows = df_m[(df_m['encoding'] == cr['encoding']) & (df_m['n_features'] == cr['n_features'])]
                    color    = color_cycle[ci % 10]
                    enc_label = f"{cr['encoding']} n={int(cr['n_features'])}"
                    all_y_true, all_y_prob = [], []
                    for _, row in enc_rows.iterrows():
                        try:
                            with sqlite3.connect(RESULT_DB, timeout=30) as conn_r:
                                preds_row = pd.read_sql(
                                    "SELECT y_true, y_prob FROM predictions WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                                    conn_r, params=(int(row['fold']), row['encoding'], int(row['n_features']), model_name)
                                )
                            all_y_true.extend(preds_row['y_true'].tolist())
                            all_y_prob.extend(preds_row['y_prob'].tolist())
                        except Exception:
                            continue
                    if all_y_true:
                        from sklearn.calibration import calibration_curve
                        try:
                            frac_pos, mean_pred = calibration_curve(all_y_true, all_y_prob, n_bins=10)
                            ax3.plot(mean_pred, frac_pos, 'o-', color=color, linewidth=1.5,
                                     markersize=4, label=enc_label)
                        except Exception:
                            pass
                ax3.set_xlabel('Mean Predicted Probability', fontsize=11)
                ax3.set_ylabel('Fraction of Positives', fontsize=11)
                ax3.legend(loc='upper left', fontsize=8)
                ax3.grid(alpha=0.3)
                plt.tight_layout(); pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)

                # Page 4: Clinical threshold table
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                ax4.set_title(f'{model_name.upper()} -- Clinical Threshold Table (threshold={CLINICAL_THRESHOLD})', fontweight='bold')
                ax4.axis('off')
                col_labels = ['Encoding', 'n', 'Fold', 'N_flagged', 'N_caught', 'N_missed', 'Precision', 'Recall']
                table_data = []
                for _, row in df_m.sort_values(['encoding', 'n_features', 'fold']).iterrows():
                    n_missed = int(row['n_total_deviations']) - int(row['n_caught'])
                    n_str    = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    table_data.append([
                        row['encoding'], n_str, str(int(row['fold'])),
                        f"{int(row['n_flagged']):,}", f"{int(row['n_caught']):,}",
                        f"{n_missed:,}",
                        f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                    ])
                if table_data:
                    tbl = ax4.table(
                        cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center'
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(8)
                    tbl.auto_set_column_width(col=list(range(len(col_labels))))
                plt.tight_layout(); pdf.savefig(fig4, bbox_inches='tight'); plt.close(fig4)

            print(f"   task3_{model_name}.pdf  -> {pdf_path}  (4 pages)")

        # -- Main fold loop ----------------------------------------------------
        sep("MAIN LOOP -- fold -> load encodings -> model")
        all_metrics   = []
        model_buckets = {m: [] for m in MODEL_LIST}

        for fold in all_folds:
            train_idx, val_idx = fold_indices[fold]
            fold_label = f"FOLD {fold}  train={len(train_idx):,}  val={len(val_idx):,}" + (' (temporal)' if fold == 5 else '')
            sep(fold_label, char='-')

            y_train = load_target(fold, 'train')
            y_val   = load_target(fold, 'val')
            n_pos_tr = y_train.sum(); n_pos_va = y_val.sum()
            print(f"  y_train: n={len(y_train):,}  pos={n_pos_tr:.0f}  rate={n_pos_tr/len(y_train)*100:.2f}%")
            print(f"  y_val  : n={len(y_val):,}  pos={n_pos_va:.0f}  rate={n_pos_va/len(y_val)*100:.2f}%")

            # Dynamic scale_pos_weight for XGBoost per fold
            xgb_scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
            print(f"  XGB scale_pos_weight (fold-dynamic): {xgb_scale_pos_weight:.2f}")

            # Load all matrices for this fold
            print(f"\n  Loading all matrices for fold {fold} (one DB connection)...")
            raw_mats = load_fold_matrices(fold)

            fold_encoded = {}
            for (split, enc, n), arr in raw_mats.items():
                if split == 'train':
                    fold_encoded.setdefault((enc, n), {})['train'] = arr
                else:
                    fold_encoded.setdefault((enc, n), {})['val'] = arr

            # Scale each combo (MinMaxScaler fit on train only)
            for key in list(fold_encoded.keys()):
                d = fold_encoded[key]
                if 'train' not in d or 'val' not in d:
                    del fold_encoded[key]; continue
                scaler  = MinMaxScaler()
                X_tr_sc = scaler.fit_transform(d['train'])
                X_va_sc = scaler.transform(d['val'])
                fold_encoded[key] = (d['train'], X_tr_sc, d['val'], X_va_sc)

            # Report loaded combos
            for enc, n in sorted(fold_encoded.keys(), key=lambda x: (x[0], x[1])):
                X_tr, _, X_va, _ = fold_encoded[(enc, n)]
                print(f"    {enc:<28} n={n:>3}  train={X_tr.shape}  val={X_va.shape}")

            all_combos = sorted(fold_encoded.keys(), key=lambda x: (x[0], x[1]))

            for (encoding, n) in all_combos:
                X_train, X_train_sc, X_val, X_val_sc = fold_encoded[(encoding, n)]
                n_label = f"n={n}" if n > 0 else "struct only"
                print(f"\n  {'-'*60}")
                print(f"  Encoding: {encoding.upper()}  {n_label}")

                for model_name in MODEL_LIST:
                    if resume_mode and (fold, encoding, n, model_name) in done_combos:
                        print(f"    [>>]  {model_name} -- already in result_task3.db, skipping.")
                        with sqlite3.connect(RESULT_DB, timeout=30) as _conn:
                            _saved = pd.read_sql(
                                "SELECT * FROM metrics WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                                _conn, params=(int(fold), encoding, int(n), model_name)
                            )
                        for _, _r in _saved.iterrows():
                            _d = _r.to_dict()
                            all_metrics.append(_d)
                            model_buckets[model_name].append(_d)
                        continue

                    delete_existing(fold, encoding, n, model_name)
                    print(f"     {model_name}")
                    try:
                        best_p = {}
                        # Optuna tuning (all models -- no default/skip logic)
                        idx_all  = np.arange(len(y_train))
                        idx_tr2, idx_tune = train_test_split(
                            idx_all, test_size=TUNE_SET_FRACTION, random_state=RANDOM_STATE
                        )
                        X_tr2,  X_tune  = X_train[idx_tr2],  X_train[idx_tune]
                        y_tr2,  y_tune  = y_train[idx_tr2],  y_train[idx_tune]
                        sc_inner   = MinMaxScaler()
                        X_tr2_sc   = sc_inner.fit_transform(X_tr2)
                        X_tune_sc  = sc_inner.transform(X_tune)

                        # Linear models are slow with class_weight='balanced' on large datasets.
                        # Use a small Optuna subset + fewer trials for ridge/lasso/elasticnet.
                        _LINEAR = ('ridge',)
                        if model_name in _LINEAR and LINEAR_OPTUNA_SUBSET and len(y_tr2) > LINEAR_OPTUNA_SUBSET:
                            _rng     = np.random.default_rng(RANDOM_STATE)
                            _lin_idx = _rng.choice(len(y_tr2), LINEAR_OPTUNA_SUBSET, replace=False)
                            X_tr2_opt, X_tr2_sc_opt = X_tr2[_lin_idx], X_tr2_sc[_lin_idx]
                            y_tr2_opt = y_tr2[_lin_idx]
                            n_trials_this = LINEAR_N_TRIALS
                        else:
                            X_tr2_opt, X_tr2_sc_opt, y_tr2_opt = X_tr2, X_tr2_sc, y_tr2
                            n_trials_this = N_TRIALS

                        sampler = optuna.samplers.TPESampler(
                            n_startup_trials=min(OPTUNA_N_STARTUP_TRIALS, n_trials_this - 1),
                            multivariate=True,
                            seed=RANDOM_STATE,
                        )
                        pruner = optuna.pruners.MedianPruner(
                            n_startup_trials=min(OPTUNA_N_STARTUP_TRIALS, n_trials_this - 1),
                            n_warmup_steps=3,
                        )
                        # Optuna minimises -AUC-ROC (direction='maximize' equivalent)
                        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
                        study.optimize(
                            make_objective(model_name, X_tr2_opt, X_tr2_sc_opt, X_tune, X_tune_sc, y_tr2_opt, y_tune),
                            n_trials=n_trials_this
                        )
                        best_p = study.best_trial.params
                        if model_name == 'mlp':
                            best_p['input_dim'] = X_train_sc.shape[1]
                            tf.keras.backend.clear_session()
                        elif model_name == 'xgboost':
                            best_p['scale_pos_weight'] = xgb_scale_pos_weight
                        best_auc_str = f"{-study.best_trial.value:.4f}"
                        print(f"      Optuna best AUC-ROC={best_auc_str}  params={best_p}")

                        mdl = build_model(model_name, best_p)
                        y_prob, train_time_s, infer_time_s = fit_predict(
                            model_name, mdl,
                            X_train, X_train_sc, X_val, X_val_sc,
                            y_train, yva=y_val, final=True,
                            scale_pos_weight=xgb_scale_pos_weight
                        )

                        metrics = compute_metrics(y_val, y_prob)
                        metrics.update({
                            'fold':         fold,
                            'encoding':     encoding,
                            'n_features':   n,
                            'model':        model_name,
                            'train_time_s': round(train_time_s, 4),
                            'infer_time_s': round(infer_time_s, 4),
                        })
                        all_metrics.append(metrics)
                        model_buckets[model_name].append(metrics)
                        print(f"      [OK] AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  "
                              f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
                              f"Brier={metrics['brier']:.4f}  "
                              f"Flagged={metrics['n_flagged']:,}  Caught={metrics['n_caught']:,}  "
                              f"train={train_time_s:.2f}s  infer={infer_time_s:.4f}s")

                        save_db(pd.DataFrame([metrics]), 'metrics')

                        val_case_ids = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['case_id'].values
                        pred_df = pd.DataFrame({
                            'fold':       fold,
                            'encoding':   encoding,
                            'n_features': n,
                            'model':      model_name,
                            'row_index':  val_idx,
                            'case_id':    val_case_ids[:len(y_prob)] if len(val_case_ids) >= len(y_prob) else np.full(len(y_prob), np.nan),
                            'y_true':     y_val.astype(int),
                            'y_prob':     y_prob,
                        })
                        save_db(pred_df, 'predictions')

                        # Feature importance
                        imps = {}
                        try:
                            if model_name in ('randomforest', 'xgboost', 'lightgbm'):
                                raw_imp = mdl.feature_importances_
                                top_n   = min(50, len(raw_imp))
                                ranks   = np.argsort(raw_imp)[::-1][:top_n]
                                imps    = {f'f{r}': float(raw_imp[r]) for r in ranks}
                            elif model_name in ('ridge', 'lasso', 'elasticnet'):
                                coefs = mdl.coef_[0]
                                top_n = min(50, len(coefs))
                                ranks = np.argsort(np.abs(coefs))[::-1][:top_n]
                                imps  = {f'f{r}': float(coefs[r]) for r in ranks}
                            elif model_name == 'mlp':
                                X_tf = tf.constant(X_val_sc.astype(np.float32))
                                with tf.GradientTape() as tape:
                                    tape.watch(X_tf)
                                    out = tf.cast(mdl(X_tf, training=False), tf.float32)
                                grads = tape.gradient(out, X_tf)
                                if grads is not None:
                                    mean_abs = np.abs(grads.numpy()).mean(axis=0)
                                    top_n = min(50, len(mean_abs))
                                    ranks = np.argsort(mean_abs)[::-1][:top_n]
                                    imps  = {f'f{r}': float(mean_abs[r]) for r in ranks}
                        except Exception as e:
                            print(f"      [!] Feature importance failed: {e}")

                        fi_rows = [{'fold': fold, 'encoding': encoding, 'n_features': n,
                                    'model': model_name, 'feature_rank': rank,
                                    'importance': val}
                                   for rank, (feat, val) in enumerate(
                                       sorted(imps.items(), key=lambda x: abs(x[1]), reverse=True)
                                   )]
                        if fi_rows:
                            save_db(pd.DataFrame(fi_rows), 'feature_importance')

                        params_clean = {
                            k: (int(v) if isinstance(v, np.integer)
                                else float(v) if isinstance(v, np.floating)
                                else v)
                            for k, v in best_p.items()
                        }
                        hp_rows = [{'fold': fold, 'encoding': encoding, 'n_features': n,
                                    'model': model_name, 'key': k, 'value': str(v)}
                                   for k, v in params_clean.items()]
                        if hp_rows:
                            save_db(pd.DataFrame(hp_rows), 'hyperparameter')

                    except Exception as e:
                        import traceback
                        print(f"       Error in {model_name}: {e}")
                        traceback.print_exc()

        # -- Per-model .log and .pdf -------------------------------------------
        sep("PER-MODEL ARTIFACTS  (.log + .pdf)")
        for model_name in MODEL_LIST:
            rows = model_buckets[model_name]
            if rows:
                print(f"\n  Generating artifacts for: {model_name}")
                save_model_artifacts(model_name, rows)
            else:
                print(f"\n  [!] No results for {model_name} -- skipping artifacts.")

        # -- Final aggregated summary ------------------------------------------
        sep("FINAL AGGREGATED SUMMARY  (mean +/- std across folds)")
        if all_metrics:
            df_m    = pd.DataFrame(all_metrics)
            grouped = df_m.groupby(['encoding', 'n_features', 'model'])[
                ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'brier',
                 'n_flagged', 'n_caught', 'train_time_s', 'infer_time_s']
            ].agg(['mean', 'std'])
            grouped.columns = ['_'.join(c) for c in grouped.columns]
            grouped = grouped.reset_index().sort_values('auc_roc_mean', ascending=False)
            print(f"\n  {'Encoding':<22} {'n':>5} {'Model':<14} {'AUC-ROC':>9} {'AUC-PR':>9} "
                  f"{'F1':>7} {'Brier':>7} {'Train(s)':>10} {'Infer(s)':>10}")
            print(f"  {'-'*100}")
            for _, row in grouped.iterrows():
                n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                print(f"  {row['encoding']:<22} {n_str:>5} {row['model']:<14} "
                      f"{row['auc_roc_mean']:>7.4f}+/-{row['auc_roc_std']:.4f} "
                      f"{row['auc_pr_mean']:>7.4f}+/-{row['auc_pr_std']:.4f} "
                      f"{row['f1_mean']:>6.4f} "
                      f"{row['brier_mean']:>6.4f} "
                      f"{row['train_time_s_mean']:>10.2f} "
                      f"{row['infer_time_s_mean']:>10.4f}")

            sep("BEST CONFIGURATION PER METRIC")
            for metric, direction in [
                ('auc_roc_mean',    'max'),
                ('auc_pr_mean',     'max'),
                ('f1_mean',         'max'),
                ('brier_mean',      'min'),
                ('train_time_s_mean', 'min'),
            ]:
                try:
                    idx  = grouped[metric].idxmax() if direction == 'max' else grouped[metric].idxmin()
                    best = grouped.loc[idx]
                    n_str = str(int(best['n_features'])) if best['n_features'] > 0 else 'struct_only'
                    print(f"  Best {metric:<26}: encoding={best['encoding']}  n={n_str}  model={best['model']}  value={best[metric]:.4f}")
                except Exception:
                    pass

        print(f"\n  Results -> {RESULT_DB}")
        print(f"  Log     -> {LOG_DIR}/task3_04_modeling.log")
        print(f"  [OK] Stage 04 complete.")

    finally:
        tee.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("pipeline_task3.py  --  Task 3: Surgical Procedure Deviation Detection")
    print()
    print("  Stages 01-03 auto-skip if already complete.")
    print("  Stage 04 automatically runs ALL models -- no interactive prompts.")
    print("  Stage 04 resumes automatically -- skips completed combos.")
    print("  All settings are in the CONFIG block at the top of this file.")
    print()
    print("  [1]  Stage 01 -- Pre-processing     CSV -> task3_clean.db + fold indices (incl. temporal fold 5)")
    print("  [2]  Stage 02 -- BERT Cache          full embeddings -> task3_*.npy  (skips if exists)")
    print("  [3]  Stage 03 -- Fold Encoding       impute -> ICD-10 OHE -> one-hot -> BERT PCA + RAG -> fold_encoded_task3.db")
    print("  [4]  Stage 04 -- Modeling            tune -> fit -> evaluate (all models, all n) -> result_task3.db")
    print("  [0]  Run ALL stages in order  [default]")
    print()
    # Run all stages automatically without prompting
    STAGE_MAP = {1: run_stage01, 2: run_stage02, 3: run_stage03, 4: run_stage04}
    selected = [1, 2, 3, 4]
    print(f"  Running all stages: {selected}\n")
    for s in selected:
        STAGE_MAP[s]()
    sep("ALL STAGES COMPLETE  (Task 3)")

if __name__ == '__main__':
    main()

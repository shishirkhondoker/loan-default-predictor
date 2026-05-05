from pathlib import Path

RANDOM_SEED = 42
N_SAMPLE = 50_000

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
EXPLANATIONS_DIR = ROOT_DIR / "explanations"

DEFAULT_STATUSES = ["Charged Off", "Default", "Late (31-120 days)"]

STATIC_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
]

BEHAVIORAL_FEATURES = [
    "credit_history_months",
    "loan_to_income",
    "revol_bal_to_income",
    "income_per_installment",
    "high_dti_flag",
]
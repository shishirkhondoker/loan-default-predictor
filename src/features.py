import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple derived features available around application time."""
    df = df.copy()

    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")

    df["credit_history_months"] = (
        (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
        + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    )

    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
    df["income_per_installment"] = df["annual_inc"] / (df["installment"] + 1)
    df["high_dti_flag"] = (df["dti"] > 30).astype(int)

    return df

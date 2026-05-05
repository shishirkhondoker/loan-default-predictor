import pandas as pd

from src.config import RAW_DATA_PATH, OUTPUTS_DIR


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading first few rows for column inspection...")

    df_head = pd.read_csv(RAW_DATA_PATH, nrows=5, low_memory=False)

    print(f"Total columns: {len(df_head.columns)}")

    column_list = pd.DataFrame({
        "column_number": range(1, len(df_head.columns) + 1),
        "column_name": df_head.columns
    })

    column_list.to_csv(OUTPUTS_DIR / "column_list.csv", index=False)

    print("Saved column list to outputs/column_list.csv")

    print("Reading 50,000 rows for summary...")

    df_sample = pd.read_csv(RAW_DATA_PATH, nrows=50000, low_memory=False)

    summary = pd.DataFrame({
        "column": df_sample.columns,
        "dtype": df_sample.dtypes.astype(str),
        "missing_count": df_sample.isna().sum().values,
        "missing_percent": (df_sample.isna().mean().values * 100).round(2),
        "unique_values": df_sample.nunique(dropna=True).values
    })

    summary.to_csv(OUTPUTS_DIR / "data_summary.csv", index=False)

    print("Saved data summary to outputs/data_summary.csv")

    if "loan_status" in df_sample.columns:
        loan_status_counts = df_sample["loan_status"].value_counts(dropna=False)
        loan_status_counts.to_csv(OUTPUTS_DIR / "loan_status_counts.csv")

        print("Saved loan status counts to outputs/loan_status_counts.csv")


if __name__ == "__main__":
    main()
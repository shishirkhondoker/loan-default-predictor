import pandas as pd

from src.config import RAW_DATA_PATH, N_SAMPLE, RANDOM_SEED, DEFAULT_STATUSES


def validate_target_definition(df):
    """
    Validate target definition after dropping Current loans and creating target.
    """

    print("\n================ Target Validation ================")

    print("\nLoan status distribution after dropping Current loans:")
    print(df["loan_status"].value_counts(dropna=False))

    print("\nCurrent loans remaining:")
    print((df["loan_status"] == "Current").sum())

    print("\nTarget count:")
    print(df["target"].value_counts())

    print("\nTarget percentage:")
    print((df["target"].value_counts(normalize=True) * 100).round(2))

    print("\nTarget mapping by loan_status:")
    print(df.groupby("loan_status")["target"].mean().sort_values(ascending=False))

    print("===================================================\n")


def run_sanity_checks(df):
    """
    Run basic data structure and sanity checks.
    """

    print("\n================ Data Sanity Checks ================")

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    duplicate_rows = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_rows}")

    if "borrower_id" in df.columns:
        duplicate_borrower_ids = df["borrower_id"].duplicated().sum()
        unique_borrower_ids = df["borrower_id"].nunique(dropna=True)
        missing_borrower_ids = df["borrower_id"].isna().sum()

        print(f"Unique borrower_id values: {unique_borrower_ids}")
        print(f"Duplicate borrower_id values: {duplicate_borrower_ids}")
        print(f"Missing borrower_id values: {missing_borrower_ids}")
    else:
        print("borrower_id column not found at sanity check stage.")

    numeric_non_negative_cols = [
        "loan_amnt",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "fico_range_low",
        "fico_range_high",
    ]

    print("\nNegative value checks:")

    for col in numeric_non_negative_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            print(f"{col}: {negative_count} negative values")

    if "annual_inc" in df.columns:
        zero_or_negative_income = (df["annual_inc"] <= 0).sum()
        print(f"\nAnnual income <= 0: {zero_or_negative_income}")

    if "loan_amnt" in df.columns:
        zero_or_negative_loan = (df["loan_amnt"] <= 0).sum()
        print(f"Loan amount <= 0: {zero_or_negative_loan}")

    if "dti" in df.columns:
        very_high_dti = (df["dti"] > 100).sum()
        missing_dti = df["dti"].isna().sum()
        print(f"DTI > 100: {very_high_dti}")
        print(f"Missing DTI values: {missing_dti}")

    if "revol_util" in df.columns:
        very_high_revol_util = (df["revol_util"] > 150).sum()
        missing_revol_util = df["revol_util"].isna().sum()
        print(f"Revolving utilization > 150: {very_high_revol_util}")
        print(f"Missing revol_util values: {missing_revol_util}")

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        invalid_fico = (df["fico_range_low"] > df["fico_range_high"]).sum()
        missing_fico_low = df["fico_range_low"].isna().sum()
        missing_fico_high = df["fico_range_high"].isna().sum()

        print(f"Invalid FICO range low > high: {invalid_fico}")
        print(f"Missing fico_range_low values: {missing_fico_low}")
        print(f"Missing fico_range_high values: {missing_fico_high}")

    print("====================================================\n")


def load_and_prepare_data():
    """
    Load Lending Club data in chunks to avoid memory error.
    Drop Current loans, create target, validate target definition,
    create borrower_id, run sanity checks, and return 50,000 sampled rows.
    """

    print("Reading dataset in chunks. This may take a few minutes...")

    use_columns = [
        "id",
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
        "issue_d",
        "loan_status",
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

    sampled_chunks = []
    chunksize = 100_000
    per_chunk_sample = 4_000

    reader = pd.read_csv(
        RAW_DATA_PATH,
        usecols=lambda col: col in use_columns,
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        print(f"Processing chunk {i}...")

        # Drop Current loans because their final outcome is unknown
        chunk = chunk[chunk["loan_status"] != "Current"].copy()

        if len(chunk) == 0:
            continue

        # Take a small sample from each chunk
        n = min(per_chunk_sample, len(chunk))
        chunk_sample = chunk.sample(n=n, random_state=RANDOM_SEED + i)

        sampled_chunks.append(chunk_sample)

    df = pd.concat(sampled_chunks, ignore_index=True)

    print(f"Rows collected before final sampling: {df.shape[0]}")

    # Final sample of 50,000 rows
    if len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=RANDOM_SEED)

    # Create target variable
    df["target"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)

    # Validate target definition
    validate_target_definition(df)

    # Create borrower_id for query interface
    if "id" in df.columns:
        df["borrower_id"] = df["id"].astype(str)
    else:
        df["borrower_id"] = df.index.astype(str)

    df = df.reset_index(drop=True)

    # Run data sanity checks
    run_sanity_checks(df)

    print(f"Final sample shape: {df.shape}")
    print(f"Default rate: {df['target'].mean():.2%}")

    return df
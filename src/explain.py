import pandas as pd
from src.config import OUTPUTS_DIR, EXPLANATIONS_DIR


def create_simple_explanations():
    """Create simple plain-English explanations for top 3 risky borrowers."""
    EXPLANATIONS_DIR.mkdir(exist_ok=True)
    preds = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv")
    top3 = preds.head(3)

    median_dti = preds["dti"].median()
    median_income = preds["annual_inc"].median()

    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        text = f"""# Borrower {row['borrower_id']} — predicted default risk: {row['predicted_risk']:.1%}

Top reasons:
- Debt-to-income ratio is {row['dti']:.1f}%, compared with portfolio median of {median_dti:.1f}%.
- Interest rate is {row['int_rate']}, which may indicate the lender already assessed elevated risk.
- Loan amount is ${row['loan_amnt']:,.0f} compared with annual income of ${row['annual_inc']:,.0f}.
- Income level is compared against portfolio median income of ${median_income:,.0f}.

Note: This is a simple explanation template. For a stronger submission, add SHAP-based feature contributions.
"""
        with open(EXPLANATIONS_DIR / f"borrower_{i}.md", "w") as f:
            f.write(text)


if __name__ == "__main__":
    create_simple_explanations()

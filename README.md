# Mini Loan Default Predictor

Periscope Labs AI/ML Engineer take-home assignment.

This project builds a small loan default risk prediction system using Lending Club accepted loans data. The goal is to predict which borrowers are more likely to default, explain high-risk predictions in plain English, and provide a small Streamlit query interface.

---

## Project Overview

A bank wants to identify borrowers who may default before the default actually happens. Early identification can help credit officers contact borrowers, restructure loans, or take other preventive actions.

This project includes:

- Data understanding and leakage audit
- Target creation from `loan_status`
- Safe application-time feature selection
- Two machine learning models:
  - Model 1: Static features only
  - Model 2: Static + behavioral/derived features
- Model evaluation
- Plain-English borrower explanations
- A small Streamlit query interface

---

## Dataset

This project uses the Lending Club accepted loans dataset.

Dataset file here: https://www.kaggle.com/datasets/wordsforthewise/lending-club

```text
data/raw/accepted_2007_to_2018Q4.csv
```

The original dataset is large, so the code reads it in chunks and samples 50,000 rows for faster local training.

Loans with `loan_status = Current` are removed because their final repayment outcome is unknown.

---

## Target Definition

The binary target variable is created from `loan_status`.

A loan is labeled as defaulted if `loan_status` is one of:

- `Charged Off`
- `Default`
- `Late (31-120 days)`

Target meaning:

```text
target = 1  -> default / risky borrower
target = 0  -> non-default borrower
```

Final sampled dataset:

```text
Sample size: 50,000 rows
Default rate: 21.98%
Non-default rate: 78.02%
```

---

## Project Structure

```text
loan-default-predictor/
├── data/
│   ├── raw/
│       └── accepted_2007_to_2018Q4.csv
│
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── time_split_check.py
│   └── explain.py
├── models/
│   ├── model_static.joblib
│   └── model_behavioral.joblib
├── outputs/
│   ├── metrics.csv
│   ├── confusion_matrices.txt
│   ├── test_predictions.csv
│   ├── column_list.csv
│   ├── data_summary.csv
│   ├── loan_status_count.csv
│   ├── time_split_confusion_matrix.txt
│   ├── time_split_matrix.csv
│   └── top_risk_borrowers.csv
├── explanations/
│   ├── borrower_1.md
│   ├── borrower_2.md
│   └── borrower_3.md
├── app.py
├── README.md
├── data_audit.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

### 1. Create and activate a virtual environment

For Windows Git Bash:

```bash
python -m venv venv
source venv/Scripts/activate
```

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 3. Add the dataset

Download the Lending Club accepted loans file from Kaggle and place it here:https://www.kaggle.com/datasets/wordsforthewise/lending-club

```text
data/raw/accepted_2007_to_2018Q4.csv
```

---

### 4. Train the models

```bash
python -m src.train
```

This creates trained model files and output files in:

```text
models/
outputs/
```

---

### 5. Generate borrower explanations

```bash
python -m src.explain
```

This creates explanation files in:

```text
explanations/
```

---

### 6. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Streamlit Query Interface

The Streamlit app supports three simple query types.

### Query 1: Show top 10 highest-risk borrowers

Example:

```text
Show me the top 10 highest-risk borrowers in the test set.
```

---

### Query 2: Explain why a borrower was flagged

Example:

```text
Why was borrower 12345 flagged?
```

Replace `12345` with an actual borrower ID from the prediction table.

---

### Query 3: Average default rate for loans above $20,000

Example:

```text
What's the average default rate for loans above $20,000?
```

In my run, this query returned:

```text
24.95%
```

---

## Key Decisions

### 1. Dropping Current loans

I removed loans with `loan_status = Current` because their final repayment outcome is unknown. Keeping them would make the target unreliable.

---

### 2. Avoiding target leakage

The Lending Club dataset contains many columns that are only known after the loan has started performing or after the borrower has defaulted. I excluded these columns because they would leak the target.

Examples of excluded leakage columns:

- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`
- `total_rec_int`
- `total_rec_late_fee`
- `recoveries`
- `collection_recovery_fee`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `next_pymnt_d`
- `out_prncp`
- `out_prncp_inv`
- `last_fico_range_high`
- `last_fico_range_low`
- `hardship_status`
- `settlement_status`
- `settlement_amount`

A more detailed leakage review is included in `data_audit.md`.

---

### 3. Sampling 50,000 rows

The full dataset is large and caused local memory pressure. To keep the project runnable on a normal laptop, I sampled 50,000 rows using a fixed random seed.

The data loading code reads the CSV in chunks and samples from each chunk before creating the final training sample.

---

### 4. Feature selection

I used features that would plausibly be available at application or approval time.

Important features include:

- Loan amount
- Loan term
- Interest rate
- Monthly installment
- Grade and sub-grade
- Employment length
- Home ownership
- Annual income
- Verification status
- Loan purpose
- Debt-to-income ratio
- FICO score range
- Recent credit inquiries
- Open accounts
- Public records
- Revolving balance
- Revolving utilization
- Total accounts
- Application type

---

### 5. Two-model comparison

I trained two models on the same stratified 80/20 train-test split.

#### Model 1: Static features only

This model uses only application-time/static borrower and loan features.

#### Model 2: Static + behavioral/derived features

This model uses the same static features plus derived features:

- `credit_history_months`
- `loan_to_income`
- `revol_bal_to_income`
- `income_per_installment`
- `high_dti_flag`

These derived features are intended to capture affordability, debt burden, and credit history strength.

---

### 6. Evaluation metrics

I did not rely on accuracy because the dataset is imbalanced.

The models were evaluated using:

- AUC-ROC
- Precision @ top 10%
- Recall @ top 10%
- Confusion matrix at 0.5 threshold
- Confusion matrix at the threshold that maximizes F1

Precision and recall at the top 10% are useful because a bank may only have capacity to review the riskiest borrowers first.

---

## Preprocessing

The preprocessing pipeline includes:

- Selecting safe application-time columns
- Dropping `Current` loans
- Creating the binary target variable
- Sampling 50,000 rows
- Median imputation for numeric features
- Most-frequent imputation for categorical features
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Derived feature engineering for Model 2

---

## Model Evaluation

Both models were trained and evaluated on the same stratified 80/20 train-test split.

| Model               | AUC-ROC | Precision @ Top 10% | Recall @ Top 10% |
| ------------------- | ------: | ------------------: | ---------------: |
| Static only         |  0.6977 |               0.469 |           0.2134 |
| Static + behavioral |  0.6979 |               0.472 |           0.2147 |

---

## Confusion Matrices

### Model 1: Static Only

At 0.5 threshold:

```text
[[4967 2835]
 [ 785 1413]]
```

At best F1 threshold:

```text
[[5709 2093]
 [ 984 1214]]
```

---

### Model 2: Static + Behavioral

At 0.5 threshold:

```text
[[4966 2836]
 [ 785 1413]]
```

At best F1 threshold:

```text
[[5634 2168]
 [ 959 1239]]
```

---

## Model Choice

I would choose the Static + behavioral model for a next-stage deployment test.

It slightly improves AUC-ROC, Precision @ Top 10%, and Recall @ Top 10% compared with the static-only model.

However, the improvement is small, so I would not overclaim the difference. The main reason I would choose it is that the derived features add intuitive borrower-level risk information without using obvious leakage columns.

Before using this model in production, I would validate it more carefully using time-based validation, probability calibration, and monitoring on newer borrower data.

---

## Explanation Layer

For the better-performing model, I selected high-risk borrowers from the top predicted-risk group and generated plain-English explanations.

The explanation files are saved in:

```text
explanations/
```

The explanations are written for a credit officer rather than a machine learning engineer. They focus on practical borrower risk factors such as:

- High debt-to-income ratio
- High loan amount relative to income
- High interest rate
- High revolving utilization
- Shorter credit history
- Recent credit inquiries
- Weak affordability indicators

---

## Limitations

This is a small take-home assignment, so I intentionally kept the scope limited.

Main limitations:

- The model uses a random train-test split instead of a time-based validation split.
- The model is trained on US Lending Club data and may not transfer directly to another country or bank.
- Model performance is moderate and not production-ready.
- Hyperparameter tuning was limited.
- Calibration was not deeply evaluated.
- The explanation layer is simplified.
- The Streamlit app uses simple query handling instead of a full natural language system.

---

## What I Would Do With More Time

With more time, I would improve the project in the following ways:

1. **Use a time-based split**  
   A random split is acceptable for this small assignment, but a time-based split would better simulate real deployment.

2. **Check model calibration**  
   I would check whether predicted probabilities match observed default rates. For example, loans predicted at 30% risk should default roughly 30% of the time.

3. **Try stronger models**  
   I would compare logistic regression with LightGBM or XGBoost and tune hyperparameters.

4. **Improve the explanation layer**  
   I would add SHAP-based explanations to better show feature-level contribution for each borrower.

5. **Evaluate subgroup stability**  
   I would check model performance across loan grade, issue year, income group, loan purpose, and other borrower segments.

6. **Review ambiguous features more deeply**  
   Some Lending Club fields may be available at different times. I would verify each feature against the data dictionary before production use.

7. **Improve the Streamlit app**  
   I would add borrower search, filters, charts, downloadable tables, and clearer explanation formatting.

8. **Add automated tests**  
   I would add tests for data loading, feature generation, metric calculation, and query handling.

9. **Validate on local bank data**  
   Since this dataset is based on US Lending Club borrowers, I would not directly deploy it on Bangladeshi borrower data. I would retrain or recalibrate the model using local borrower and repayment data.

---

## Reproducibility Notes

The project uses a fixed random seed for sampling and train-test splitting.

Expected workflow:

```bash
python -m src.train
python -m src.explain
python -m streamlit run app.py
```

The large raw dataset should not be committed to GitHub. It should be placed locally under:

```text
data/raw/
```

---

## Additional Time-Based Validation

The main assignment result uses the required stratified 80/20 split. As an additional robustness check, I also tested a time-based split using `issue_d`, where older loans were used for training and newer loans were used for testing.

This is closer to a real credit-risk deployment because a bank would train on historical loans and predict future borrowers.

The results are saved in:

```text
outputs/time_split_metrics.csv
outputs/time_split_confusion_matrix.txt
```

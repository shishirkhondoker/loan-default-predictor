# Data Audit and Leakage Review

## Dataset

This project uses the Lending Club accepted loans dataset. I used a 50,000-row sample from the accepted loan records to keep the project reproducible and fast to run locally.

The target variable was created from `loan_status`.

A loan is treated as defaulted if `loan_status` is one of:

- `Charged Off`
- `Default`
- `Late (31-120 days)`

Loans with `loan_status = Current` were removed because their final outcome is not yet known.

Final sample size: 50,000 rows  
Default rate: 21.98%  
Non-default rate: 78.02%

---

## Columns Used as Application-Time Features

I used features that would plausibly be available at loan application or approval time.

| Column                | Meaning                         | Reason for Use                            |
| --------------------- | ------------------------------- | ----------------------------------------- |
| `loan_amnt`           | Requested loan amount           | Known at application time                 |
| `term`                | Loan duration                   | Known at loan origination                 |
| `int_rate`            | Interest rate                   | Known once loan is approved               |
| `installment`         | Monthly payment amount          | Known once loan terms are set             |
| `grade`               | Lending Club loan grade         | Risk grade available at approval          |
| `sub_grade`           | More detailed grade             | Risk grade available at approval          |
| `emp_length`          | Employment length               | Borrower profile information              |
| `home_ownership`      | Rent, own, mortgage, etc.       | Borrower profile information              |
| `annual_inc`          | Annual income                   | Application-time financial information    |
| `verification_status` | Income verification status      | Application-time underwriting information |
| `purpose`             | Loan purpose                    | Known at application time                 |
| `dti`                 | Debt-to-income ratio            | Application-time affordability feature    |
| `delinq_2yrs`         | Delinquencies in last 2 years   | Credit history feature                    |
| `earliest_cr_line`    | Oldest credit line date         | Used to derive credit history length      |
| `fico_range_low`      | Lower FICO score range          | Credit score information                  |
| `fico_range_high`     | Upper FICO score range          | Credit score information                  |
| `inq_last_6mths`      | Recent credit inquiries         | Credit-seeking behavior                   |
| `open_acc`            | Number of open credit accounts  | Credit profile feature                    |
| `pub_rec`             | Public records                  | Credit risk feature                       |
| `revol_bal`           | Revolving balance               | Debt burden feature                       |
| `revol_util`          | Revolving utilization           | Credit utilization feature                |
| `total_acc`           | Total credit accounts           | Credit profile feature                    |
| `initial_list_status` | Initial listing status          | Loan origination information              |
| `application_type`    | Individual or joint application | Application type                          |

---

## Engineered Features

For the second model, I created several derived features:

| Feature                  | Meaning                                                 |
| ------------------------ | ------------------------------------------------------- |
| `credit_history_months`  | Months between earliest credit line and loan issue date |
| `loan_to_income`         | Loan amount divided by annual income                    |
| `revol_bal_to_income`    | Revolving balance divided by annual income              |
| `income_per_installment` | Annual income divided by monthly installment            |
| `high_dti_flag`          | Whether DTI is greater than 30                          |

These features were designed to capture borrower affordability, debt pressure, and credit history length.

---

## Leakage Columns Excluded

The following columns were excluded because they are known after loan origination or after repayment/default behavior is observed. Using these features would leak the target.

| Column                    | Why It Leaks                                                    |
| ------------------------- | --------------------------------------------------------------- |
| `total_pymnt`             | Total amount paid is only known after repayment behavior occurs |
| `total_pymnt_inv`         | Investor payment amount is post-origination information         |
| `total_rec_prncp`         | Principal received is repayment outcome information             |
| `total_rec_int`           | Interest received is repayment outcome information              |
| `total_rec_late_fee`      | Late fees are only known after payment behavior                 |
| `recoveries`              | Recovery amount is only known after default/collection          |
| `collection_recovery_fee` | Collection fee is only known after collection activity          |
| `last_pymnt_d`            | Last payment date is post-origination information               |
| `last_pymnt_amnt`         | Last payment amount is post-origination information             |
| `next_pymnt_d`            | Future payment schedule/status may reveal loan performance      |
| `out_prncp`               | Outstanding principal changes after repayment begins            |
| `out_prncp_inv`           | Investor outstanding principal is post-origination information  |
| `last_credit_pull_d`      | Later credit pull may happen after loan performance changes     |
| `last_fico_range_high`    | Updated FICO score may reflect later borrower behavior          |
| `last_fico_range_low`     | Updated FICO score may reflect later borrower behavior          |
| `hardship_flag`           | Hardship status is known after loan starts                      |
| `hardship_status`         | Indicates post-loan distress                                    |
| `settlement_status`       | Settlement only occurs after repayment/default problems         |
| `settlement_amount`       | Settlement amount is post-default or post-distress information  |
| `debt_settlement_flag`    | Indicates post-origination settlement activity                  |

---

## Class Imbalance

The final sampled dataset has:

- Default rate: 21.98%
- Non-default rate: 78.02%

This means the dataset is imbalanced. Accuracy alone would not be a good metric, so I evaluated the models using AUC-ROC, Precision @ Top 10%, Recall @ Top 10%, and confusion matrices.

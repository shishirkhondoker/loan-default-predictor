import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RANDOM_SEED,
    STATIC_FEATURES,
    BEHAVIORAL_FEATURES,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from src.data_prep import load_and_prepare_data
from src.features import add_behavioral_features
from src.evaluate import evaluate_model


def build_pipeline(features):
    numeric_features = [
        "loan_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "credit_history_months", "loan_to_income", "revol_bal_to_income",
        "income_per_installment", "high_dti_flag",
    ]
    numeric_features = [f for f in numeric_features if f in features]
    categorical_features = [f for f in features if f not in numeric_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data()
    df = add_behavioral_features(df)

    y = df["target"]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    y_train = df_train["target"]
    y_test = df_test["target"]

    model1_features = STATIC_FEATURES
    model2_features = STATIC_FEATURES + BEHAVIORAL_FEATURES

    model1 = build_pipeline(model1_features)
    model1.fit(df_train[model1_features], y_train)
    results1 = evaluate_model(model1, df_test[model1_features], y_test)

    model2 = build_pipeline(model2_features)
    model2.fit(df_train[model2_features], y_train)
    results2 = evaluate_model(model2, df_test[model2_features], y_test)

    joblib.dump(model1, MODELS_DIR / "model_static.joblib")
    joblib.dump(model2, MODELS_DIR / "model_behavioral.joblib")

    metrics = pd.DataFrame([
        {
            "model": "static_only",
            "auc": results1["auc"],
            "precision_top10": results1["precision_top10"],
            "recall_top10": results1["recall_top10"],
        },
        {
            "model": "static_plus_behavioral",
            "auc": results2["auc"],
            "precision_top10": results2["precision_top10"],
            "recall_top10": results2["recall_top10"],
        },
    ])
    metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    preds = df_test[["borrower_id", "loan_amnt", "annual_inc", "dti", "int_rate", "target"]].copy()
    preds["predicted_risk"] = results2["probabilities"]
    preds = preds.sort_values("predicted_risk", ascending=False)
    preds.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    preds.head(10).to_csv(OUTPUTS_DIR / "top_risk_borrowers.csv", index=False)

    with open(OUTPUTS_DIR / "confusion_matrices.txt", "w") as f:
        f.write("Model 1 confusion matrix at 0.5:\n")
        f.write(str(results1["confusion_matrix_0_5"]))
        f.write("\n\nModel 1 confusion matrix at best F1 threshold:\n")
        f.write(str(results1["confusion_matrix_best_f1"]))
        f.write("\n\nModel 2 confusion matrix at 0.5:\n")
        f.write(str(results2["confusion_matrix_0_5"]))
        f.write("\n\nModel 2 confusion matrix at best F1 threshold:\n")
        f.write(str(results2["confusion_matrix_best_f1"]))

    print(metrics)
    print("Training complete. Outputs saved.")


if __name__ == "__main__":
    main()

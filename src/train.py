import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

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


# Function to build pipeline for models
def build_pipeline(features, model_type="logistic"):
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

    # Use the selected model based on model_type
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=15, learning_rate=0.1)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    
    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load and prepare the data
    df = load_and_prepare_data()
    df = add_behavioral_features(df)

    # Target variable
    y = df["target"]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Split the data into training and test sets
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    y_train = df_train["target"]
    y_test = df_test["target"]

    # Feature sets
    model1_features = STATIC_FEATURES
    model2_features = STATIC_FEATURES + BEHAVIORAL_FEATURES

    # Logistic Regression Model (Static only)
    model_logistic = build_pipeline(model1_features, model_type="logistic")
    model_logistic.fit(df_train[model1_features], y_train)
    results_logistic = evaluate_model(model_logistic, df_test[model1_features], y_test)

    # Random Forest Model (Static + Behavioral features)
    model_rf = build_pipeline(model2_features, model_type="random_forest")
    model_rf.fit(df_train[model2_features], y_train)
    results_rf = evaluate_model(model_rf, df_test[model2_features], y_test)

    # LightGBM Model (Static + Behavioral features)
    model_lgbm = build_pipeline(model2_features, model_type="lightgbm")
    model_lgbm.fit(df_train[model2_features], y_train)
    results_lgbm = evaluate_model(model_lgbm, df_test[model2_features], y_test)

    # Save models
    joblib.dump(model_logistic, MODELS_DIR / "model_logistic.joblib")
    joblib.dump(model_rf, MODELS_DIR / "model_random_forest.joblib")
    joblib.dump(model_lgbm, MODELS_DIR / "model_lightgbm.joblib")

    # Save metrics
    metrics = pd.DataFrame([
        {
            "model": "logistic_regression",
            "auc": results_logistic["auc"],
            "precision_top10": results_logistic["precision_top10"],
            "recall_top10": results_logistic["recall_top10"],
        },
        {
            "model": "random_forest",
            "auc": results_rf["auc"],
            "precision_top10": results_rf["precision_top10"],
            "recall_top10": results_rf["recall_top10"],
        },
        {
            "model": "lightgbm",
            "auc": results_lgbm["auc"],
            "precision_top10": results_lgbm["precision_top10"],
            "recall_top10": results_lgbm["recall_top10"],
        },
    ])
    metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    # Save predictions
    preds = df_test[["borrower_id", "loan_amnt", "annual_inc", "dti", "int_rate", "target"]].copy()
    preds["predicted_risk"] = results_lgbm["probabilities"]
    preds = preds.sort_values("predicted_risk", ascending=False)
    preds.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    preds.head(10).to_csv(OUTPUTS_DIR / "top_risk_borrowers.csv", index=False)

    # Saving confusion matrices
    with open(OUTPUTS_DIR / "confusion_matrices.txt", "w") as f:
        f.write("Logistic Regression confusion matrix at 0.5:\n")
        f.write(str(results_logistic["confusion_matrix_0_5"]))
        f.write("\n\nRandom Forest confusion matrix at 0.5:\n")
        f.write(str(results_rf["confusion_matrix_0_5"]))
        f.write("\n\nLightGBM confusion matrix at 0.5:\n")
        f.write(str(results_lgbm["confusion_matrix_0_5"]))

    print(metrics)
    print("Training complete. Outputs saved.")


if __name__ == "__main__":
    main()
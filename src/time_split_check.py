import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.data_prep import load_and_prepare_data
from src.features import add_behavioral_features
from src.config import RANDOM_SEED, STATIC_FEATURES, BEHAVIORAL_FEATURES, OUTPUTS_DIR


def precision_recall_at_top_k(y_true, y_score, top_percent=0.10):
    result = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score
    })

    result = result.sort_values("y_score", ascending=False)

    top_n = int(len(result) * top_percent)
    top_group = result.head(top_n)

    precision_top_k = top_group["y_true"].mean()
    recall_top_k = top_group["y_true"].sum() / result["y_true"].sum()

    return precision_top_k, recall_top_k


def build_model(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    return model


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()
    df = add_behavioral_features(df)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).copy()

    df = df.sort_values("issue_d").reset_index(drop=True)

    features = STATIC_FEATURES + BEHAVIORAL_FEATURES

    available_features = [col for col in features if col in df.columns]

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[available_features]
    y_train = train_df["target"]

    X_test = test_df[available_features]
    y_test = test_df["target"]

    print("\n================ Time-Based Split Check ================")
    print(f"Train date range: {train_df['issue_d'].min().date()} to {train_df['issue_d'].max().date()}")
    print(f"Test date range: {test_df['issue_d'].min().date()} to {test_df['issue_d'].max().date()}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")

    model = build_model(X_train)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    precision_top10, recall_top10 = precision_recall_at_top_k(y_test, probs)

    preds_05 = (probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, preds_05)

    results = pd.DataFrame([
        {
            "split_type": "time_based",
            "model": "static_plus_behavioral",
            "auc": auc,
            "precision_top10": precision_top10,
            "recall_top10": recall_top10,
            "train_start": train_df["issue_d"].min(),
            "train_end": train_df["issue_d"].max(),
            "test_start": test_df["issue_d"].min(),
            "test_end": test_df["issue_d"].max(),
            "train_default_rate": y_train.mean(),
            "test_default_rate": y_test.mean(),
        }
    ])

    results.to_csv(OUTPUTS_DIR / "time_split_metrics.csv", index=False)

    with open(OUTPUTS_DIR / "time_split_confusion_matrix.txt", "w") as f:
        f.write("Time-based split confusion matrix at 0.5 threshold:\n")
        f.write(str(cm_05))
        f.write("\n")

    print("\nTime-based split results:")
    print(results[["model", "auc", "precision_top10", "recall_top10"]])

    print("\nConfusion matrix at 0.5 threshold:")
    print(cm_05)

    print("\nSaved:")
    print("outputs/time_split_metrics.csv")
    print("outputs/time_split_confusion_matrix.txt")
    print("========================================================\n")


if __name__ == "__main__":
    main()
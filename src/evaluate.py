import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    cutoff = np.quantile(probs, 0.90)
    top_10_pred = probs >= cutoff

    precision_top10 = y_test[top_10_pred].mean()
    recall_top10 = y_test[top_10_pred].sum() / y_test.sum()

    pred_05 = (probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, pred_05)

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1[:-1])
    best_threshold = thresholds[best_idx]
    pred_best = (probs >= best_threshold).astype(int)
    cm_best = confusion_matrix(y_test, pred_best)

    return {
        "auc": auc,
        "precision_top10": precision_top10,
        "recall_top10": recall_top10,
        "confusion_matrix_0_5": cm_05,
        "best_f1_threshold": best_threshold,
        "confusion_matrix_best_f1": cm_best,
        "probabilities": probs,
    }

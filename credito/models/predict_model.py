import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from credito.utils.paths import REPORTS_DIR

def evaluate_models(models, X_test, y_test):
    """
    Evalúa los modelos entrenados, muestra y persiste métricas reales.
    """
    print("--> Evaluando modelos...")

    results = {}
    for name, model in models.items():
        print(f"\n{'='*10} Reporte para: {name} {'='*10}")
        predictions = model.predict(X_test)
        proba_positive = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

        report = classification_report(y_test, predictions, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        roc_auc = float(roc_auc_score(y_test, proba_positive))
        precision, recall, _ = precision_recall_curve(y_test, proba_positive)
        pr_auc = float(auc(recall, precision))

        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")

        results[name] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": report.get("accuracy"),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas persistidas en {REPORTS_DIR / 'metrics.json'}")

    return results

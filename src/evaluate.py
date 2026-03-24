import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from preprocess import (
    load_data,
    feature_engineering,
    encode_categorical,
    split_data,
    scale_data
)


def evaluate_thresholds(model, X, y_true):
    thresholds = [0.3, 0.4, 0.5, 0.6]

    y_proba = model.predict_proba(X)[:, 1]

    for t in thresholds:
        print(f"\n===== Seuil = {t} =====")

        y_pred = (y_proba >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("TP (bons défauts détectés):", tp)
        print("FN (clients risqués acceptés):", fn)
        print("FP (bons clients refusés):", fp)
        print("TN (bons clients acceptés):", tn)


def evaluate_models():
    # Création du dossier pour enregistrer les figures
    os.makedirs("reports/figures", exist_ok=True)

    # ===============================
    # Préparation des données
    # ===============================
    df = load_data("data/raw/bank_credit_dataset.csv")
    df = feature_engineering(df)
    df = encode_categorical(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # ===============================
    # Chargement des modèles
    # ===============================
    rf_model = joblib.load("models/random_forest.pkl")
    log_model = joblib.load("models/logistic.pkl")
    mlp_model = joblib.load("models/mlp.pkl")

    models = {
        "Logistic_Regression": (log_model, X_test_scaled),
        "Random_Forest": (rf_model, X_test),
        "MLP": (mlp_model, X_test_scaled)
    }

    # ===============================
    # 1. Matrices de confusion
    # ===============================
    for name, (model, X) in models.items():
        y_pred = model.predict(X)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice de confusion - {name}")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig(f"reports/figures/confusion_matrix_{name}.png")
        plt.close()

    # ===============================
    # 2. Courbes ROC
    # ===============================
    plt.figure(figsize=(8, 6))

    for name, (model, X) in models.items():
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbes ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curves.png")
    plt.close()

    print("Graphiques enregistrés dans reports/figures/")

    # ===============================
    # 3. Optimisation du seuil
    # ===============================
    print("\n===== OPTIMISATION DU SEUIL - RANDOM FOREST =====")
    evaluate_thresholds(rf_model, X_test, y_test)


if __name__ == "__main__":
    evaluate_models()
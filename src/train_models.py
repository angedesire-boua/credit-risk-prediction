import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_data, feature_engineering, encode_categorical, split_data, scale_data


def train_models():
    # Chargement
    df = load_data("data/raw/bank_credit_dataset.csv")

    # Feature engineering
    df = feature_engineering(df)

    # Encodage
    df = encode_categorical(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Sauvegarde des colonnes
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

    # Scaling
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 1. Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    # 2. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 3. MLP
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=300,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    models = {
        "Logistic Regression": (log_model, X_test_scaled),
        "Random Forest": (rf_model, X_test),
        "MLP": (mlp_model, X_test_scaled)
    }

    for name, (model, X) in models.items():
        print(f"\n===== {name} =====")
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(log_model, "models/logistic.pkl")
    joblib.dump(mlp_model, "models/mlp.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModèles sauvegardés !")


if __name__ == "__main__":
    train_models()
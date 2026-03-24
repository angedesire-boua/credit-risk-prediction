import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def feature_engineering(df):
    df["DebtRatio"] = df["LoanAmount"] / (df["Income"] + 1)
    df["LoanInteraction"] = df["LoanAmount"] * df["ExistingLoans"]
    return df


def encode_categorical(df):
    categorical_cols = ["JobType", "MaritalStatus"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def split_data(df):
    X = df.drop("Default", axis=1)
    y = df["Default"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df = load_data("data/raw/bank_credit_dataset.csv")

    df = feature_engineering(df)
    df = encode_categorical(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Prétraitement OK")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
import os
import numpy as np
import pandas as pd


def generate_credit_dataset(n=5000, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(21, 70, n)
    income = np.random.normal(40000, 15000, n)
    credit_score = np.random.normal(650, 80, n)

    loan_amount = np.random.normal(15000, 7000, n)
    loan_duration = np.random.choice([12, 24, 36, 48, 60], n)

    existing_loans = np.random.poisson(1.5, n)
    balance = np.random.normal(5000, 4000, n)

    job_type = np.random.choice(
        ["employee", "self-employed", "manager", "unemployed"], n
    )

    marital_status = np.random.choice(
        ["single", "married", "divorced"], n
    )

    home_owner = np.random.choice([0, 1], n)

    # Construction d'un score de risque
    risk = (
        0.003 * loan_amount
        - 0.002 * income
        - 0.01 * credit_score
        + 0.5 * existing_loans
    )

    prob = 1 / (1 + np.exp(-risk))

    default = (prob > np.random.rand(n)).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "CreditScore": credit_score,
        "LoanAmount": loan_amount,
        "LoanDuration": loan_duration,
        "ExistingLoans": existing_loans,
        "Balance": balance,
        "JobType": job_type,
        "MaritalStatus": marital_status,
        "HomeOwner": home_owner,
        "Default": default
    })

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    df = generate_credit_dataset()
    output_path = "data/raw/bank_credit_dataset.csv"
    df.to_csv(output_path, index=False)

    print("Dataset généré avec succès.")
    print(f"Fichier sauvegardé : {output_path}")
    print(df.head())
    print("\nDimensions :", df.shape)
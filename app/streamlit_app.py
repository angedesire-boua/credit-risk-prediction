import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# CONFIGURATION DE LA PAGE
# =========================
st.set_page_config(
    page_title="Prédiction du risque de crédit",
    page_icon="💳",
    layout="wide"
)

# =========================
# CHEMINS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# =========================
# CHARGEMENT DES ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, feature_columns

model, feature_columns = load_artifacts()

# =========================
# STYLE CSS
# =========================
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 8px;
    }

    .small-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        margin-top: 10px;
    }

    .result-risk {
        background: #fff1f2;
        padding: 22px;
        border-radius: 18px;
        border-left: 7px solid #dc2626;
        margin-top: 20px;
    }

    .result-safe {
        background: #f0fdf4;
        padding: 22px;
        border-radius: 18px;
        border-left: 7px solid #16a34a;
        margin-top: 20px;
    }

    .info-card {
        background: #f8fafc;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 12px;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# EN-TÊTE
# =========================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">💳 Prédiction du risque de crédit</div>
        <div class="hero-subtitle">
            Application de prédiction du risque de crédit pour aider à la décision bancaire.
        </div>
        <div class="hero-subtitle">
            Modèle utilisé : <b>Random Forest</b> | Seuil optimisé recommandé : <b>0.40</b>
        </div>
        <div class="small-badge">🤖 Powered by Machine Learning</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Paramètres")
threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

st.sidebar.info(
    "Un seuil plus faible détecte plus de clients risqués, "
    "mais augmente aussi les faux positifs."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Conseil")
st.sidebar.write(
    "Pour une banque prudente, un seuil autour de 0.40 permet "
    "de mieux détecter les clients à risque."
)

# =========================
# FORMULAIRE
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Informations personnelles")
    age = st.number_input("Âge", min_value=18, max_value=100, value=30)
    income = st.number_input("Revenu annuel", min_value=0.0, value=40000.0, step=1000.0)
    credit_score = st.number_input("Score de crédit", min_value=300, max_value=900, value=650)
    marital_status = st.selectbox(
        "Statut matrimonial",
        ["single", "married", "divorced"],
        format_func=lambda x: {"single": "Single", "married": "Marié(e)", "divorced": "Divorcé(e)"}[x]
    )
    home_owner = st.selectbox(
        "Propriétaire ?",
        [0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non"
    )

with col2:
    st.subheader("🏦 Informations financières")
    loan_amount = st.number_input("Montant du prêt", min_value=0.0, value=15000.0, step=500.0)
    loan_duration = st.selectbox("Durée du prêt (mois)", [12, 24, 36, 48, 60], index=2)
    existing_loans = st.number_input("Nombre de crédits existants", min_value=0, value=1, step=1)
    balance = st.number_input("Solde bancaire", value=5000.0, step=500.0)
    job_type = st.selectbox(
        "Type d’emploi",
        ["employee", "self-employed", "manager", "unemployed"],
        format_func=lambda x: {
            "employee": "Employé",
            "self-employed": "Travailleur indépendant",
            "manager": "Manager",
            "unemployed": "Sans emploi"
        }[x]
    )

predict_btn = st.button("🔎 Évaluer le risque", use_container_width=True)

# =========================
# FONCTION DE PRÉPARATION
# =========================
def build_input_dataframe():
    row = {
        "Age": age,
        "Income": income,
        "CreditScore": credit_score,
        "LoanAmount": loan_amount,
        "LoanDuration": loan_duration,
        "ExistingLoans": existing_loans,
        "Balance": balance,
        "HomeOwner": home_owner,
        "JobType": job_type,
        "MaritalStatus": marital_status
    }

    df = pd.DataFrame([row])

    # Feature engineering
    df["DebtRatio"] = df["LoanAmount"] / (df["Income"] + 1)
    df["LoanInteraction"] = df["LoanAmount"] * df["ExistingLoans"]

    # Encodage
    df = pd.get_dummies(df, columns=["JobType", "MaritalStatus"], drop_first=True)

    # Ajout des colonnes manquantes
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Respect exact de l'ordre des colonnes
    df = df[feature_columns]

    return df

# =========================
# PRÉDICTION
# =========================
if predict_btn:
    input_df = build_input_dataframe()
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= threshold else 0

    # KPI
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Probabilité de défaut", f"{proba:.2%}")
    with colB:
        st.metric("Seuil appliqué", f"{threshold:.2f}")
    with colC:
        st.metric("Décision", "Risqué" if prediction == 1 else "Fiable")

    # Jauge simple
    st.subheader("📊 Niveau de risque")
    st.progress(float(proba))

    if proba < 0.30:
        st.success("Risque faible")
    elif proba < 0.60:
        st.warning("Risque modéré")
    else:
        st.error("Risque élevé")

    # Résultat principal
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-risk">
                <h3>⚠️ Client classé à risque</h3>
                <p>
                    La probabilité estimée de défaut est de <b>{proba:.2%}</b>,
                    supérieure ou égale au seuil de décision fixé à <b>{threshold:.2f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-safe">
                <h3>✅ Client classé comme fiable</h3>
                <p>
                    La probabilité estimée de défaut est de <b>{proba:.2%}</b>,
                    inférieure au seuil de décision fixé à <b>{threshold:.2f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Interprétation métier
    st.markdown('<div class="section-title">📌 Métier d’interprétation</div>', unsafe_allow_html=True)
    if proba >= threshold:
        st.write(
            "Le profil présente un niveau de risque élevé. "
            "Dans un contexte bancaire, ce client mérite une analyse approfondie "
            "avant toute décision d’octroi."
        )
    else:
        st.write(
            "Le profil présente un niveau de risque relativement faible. "
            "Le client semble plus apte à rembourser son crédit selon le modèle."
        )

    # Recommandation automatique
    st.markdown('<div class="section-title">💡 Recommandation automatique</div>', unsafe_allow_html=True)
    if proba >= threshold:
        st.warning("👉 Recommandation : refuser le crédit ou demander des garanties supplémentaires.")
    else:
        st.success("👉 Recommandation : client potentiellement éligible au crédit.")

    # Résumé du profil
    st.markdown('<div class="section-title">📋 Résumé du client</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-card">
            <b>Âge :</b> {age} ans<br>
            <b>Revenu annuel :</b> {income:,.0f}<br>
            <b>Score de crédit :</b> {credit_score}<br>
            <b>Montant du prêt :</b> {loan_amount:,.0f}<br>
            <b>Durée :</b> {loan_duration} mois<br>
            <b>Crédits existants :</b> {existing_loans}<br>
            <b>Solde bancaire :</b> {balance:,.0f}<br>
            <b>Propriétaire :</b> {"Oui" if home_owner == 1 else "Non"}<br>
            <b>Type d’emploi :</b> {job_type}<br>
            <b>Statut matrimonial :</b> {marital_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Données techniques
    st.markdown('<div class="section-title">🧾 Données utilisées pour la prédiction</div>', unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)
# 💳 Credit Risk Prediction App

## 📌 Description

Ce projet de Data Science vise à prédire le risque de défaut de paiement d’un client bancaire à partir de ses informations financières et personnelles.

L’objectif est d’aider les institutions financières à prendre des décisions éclairées lors de l’octroi de crédits en intégrant à la fois la performance du modèle et l’impact métier.

---

## 🎯 Objectifs du projet

- Réaliser une analyse exploratoire des données (EDA)
- Construire plusieurs modèles de Machine Learning
- Comparer leurs performances
- Optimiser le seuil de décision
- Intégrer une logique métier (réduction des pertes)
- Déployer une application interactive avec Streamlit

---

## 📊 Dataset

Dataset simulé représentant des clients bancaires avec les variables suivantes :

- Age
- Income (revenu annuel)
- CreditScore
- LoanAmount
- LoanDuration
- ExistingLoans
- Balance
- JobType
- MaritalStatus
- HomeOwner
- Default (variable cible)

---

## ⚙️ Feature Engineering

Création de nouvelles variables :

- **DebtRatio** = LoanAmount / Income  
- **LoanInteraction** = LoanAmount × ExistingLoans  

---

## 🧠 Modèles utilisés

- Logistic Regression
- Random Forest (modèle retenu)
- MLP (réseau de neurones)

---

## 📈 Résultats

- Accuracy ≈ 0.98 - 0.99  
- ROC-AUC ≈ 0.99  
- Excellente capacité de classification  

Le modèle Random Forest a été retenu pour sa robustesse.

---

## ⚠️ Optimisation du seuil (IMPORTANT)

Par défaut, le seuil est 0.5.  
Nous avons testé plusieurs seuils :

| Seuil | Faux négatifs (risque) | Faux positifs |

| 0.3 | 0 | 19 |
| 0.4 | 2 | 19 |
| 0.5 | 7 | 17 |
| 0.6 | 12 | 10 |

### 🎯 Conclusion métier

- Faux négatif = perte financière 💣  
- Faux positif = opportunité manquée  

Le seuil **0.4** est optimal :

- minimise les pertes
- maintient un bon équilibre

---

## 🌐 Application Streamlit

L’application permet :

- d’entrer les données d’un client  
- de prédire le risque  
- d’ajuster le seuil de décision  
- d’obtenir :
  - la probabilité de défaut  
  - une classification (fiable / risqué)  
  - une interprétation métier  
  - une recommandation automatique  

---

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## Compétences développées

- Analyse exploratoire des données (EDA)
- Machine Learning
- Feature Engineering
- Évaluation de modèles
- Optimisation du seuil (approche métier)
- Déploiement d’application avec Streamlit
- Data storytelling

---

## Perspectives d'amélioration

- Utilisation de données réelles bancaires
- Ajout d’explicabilité des modèles (SHAP, LIME)
- Déploiement cloud (Streamlit Cloud, AWS)
- Création d’une API (FastAPI)
- Ajout de monitoring du modèle

---

## Auteur

Ange Desire Boua
🎓 Master Big Data & Intelligence Artificiellle
🚀 Aspiring Data Scientist

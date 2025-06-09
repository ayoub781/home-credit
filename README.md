# 🏠 Projet Home Credit - Scoring de Crédit

Ce projet vise à prédire la probabilité de remboursement d’un crédit par un client, à l’aide de données issues du challenge [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

## 📁 Structure du projet

- `Data/` : données brutes, intermédiaires et finales.
- `Notebook/` : tous les notebooks exploratoires et d'entraînement.
- `Scripts/` : les scripts Python de prédiction (dont Streamlit).
- `requirements.txt` : dépendances pour exécuter le projet.
## 🔎 Objectifs du projet

-Prédire si un client est fiable ou risqué pour le remboursement.
-Fournir une explication visuelle des décisions du modèle grâce à SHAP.
-Afficher la probabilité estimée par l’IA pour chaque classe, afin de rendre la prédiction plus transparente.


## 🧠 Comment fonctionne la prédiction ?

Lorsqu’un utilisateur entre les informations du client, le modèle calcule :

🔵 La probabilité que ce client rembourse (classe 0)
🔴 La probabilité qu’il ne rembourse pas (classe 1)

En fonction de la plus haute probabilité, le modèle affiche le verdict :
    🟢 Le client est considéré comme fiable pour le remboursement.
    🔴 Le client est considéré comme risqué pour le remboursement.

## 🚀 Déploiement

Une application Streamlit permet de simuler un scoring de crédit en ligne avec explication des résultats via SHAP.

## 🛠️ Technologies

- Python, Pandas, LightGBM
- SHAP pour l'interprétabilité
- Streamlit pour le front-end

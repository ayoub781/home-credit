

# 1. Importer les bibliothèques nécessaires

from preprocessing import preprocess_user_input
import streamlit as st          # Pour créer l’interface web
import pandas as pd             # Pour créer un DataFrame avec les données saisies
import joblib                   # Pour charger ton modèle .pkl
import shap
import matplotlib.pyplot as plt


# 2. Titre de l'application dans l’interface
st.title("Prédiction de remboursement de crédit")
st.write("Veuillez remplir les informations suivantes :")

#3  charges le modele


model = joblib.load("Scripts/model_final_203features.pkl")


# 4. Interface utilisateur – on crée les champs à remplir
# ⚠️ Tu adapteras les noms selon les colonnes d’entrée de ton modèle

# Création des champs du formulaire avec les 10 variables les plus importantes
ext_source_1 = st.slider("Score externe 1", 0.0, 1.0, 0.5)
ext_source_3 = st.slider("Score externe 3", 0.0, 1.0, 0.5)
ext_source_2 = st.slider("Score externe 2", 0.0, 1.0, 0.5)
amt_credit = st.number_input("Montant du crédit", min_value=0.0, value=500000.0, step=10000.0)
days_birth = st.number_input("Age du client", min_value=5000, max_value=30000, value=12000)
amt_annuity = st.number_input("Montant des annuités", min_value=0.0, value=25000.0, step=1000.0)
days_employed = st.number_input("Nombre de jours travaillés", min_value=0, max_value=15000, value=3000) #≈ 41 ans de travail
inst_amt_payment_sum = st.number_input("Somme totale des paiements mensuels des précédents crédits", min_value=0.0, value=100000.0)
amt_goods_price = st.number_input("Prix du bien acheté", min_value=0.0, value=300000.0, step=10000.0)
inst_payment_diff_mean = st.number_input("Différence moyenne entre paiement prévu et réel", min_value=0.0, value=0.0)

# Lorsque l'utilisateur clique sur le bouton :
if st.button("Prédire"):

    # Charger le template complet à 203 colonnes
    template_df = pd.read_csv("Scripts/data_template.csv")

    # Remplacer les 10 colonnes saisies par l'utilisateur
    template_df['ext_source_1'] = ext_source_1
    template_df['ext_source_3'] = ext_source_3
    template_df['ext_source_2'] = ext_source_2
    template_df['amt_credit'] = amt_credit
    template_df['days_birth'] = days_birth
    template_df['amt_annuity'] = amt_annuity
    template_df['days_employed'] = days_employed
    template_df['inst_amt_payment_sum'] = inst_amt_payment_sum
    template_df['amt_goods_price'] = amt_goods_price
    template_df['inst_payment_diff_mean'] = inst_payment_diff_mean

    # Appliquer le preprocessing si nécessaire
    input_df = preprocess_user_input(template_df)

    # Prédiction
    prediction = model.predict(input_df)
    
        # Prédiction + probabilité
    proba = model.predict_proba(input_df)[0]  # [proba_0, proba_1]
    score_fiable = proba[0]
    score_risque = proba[1]

    # Affichage du résultat
    if prediction[0] == 0:
        st.success("🟢 Le client est considéré comme **fiable** pour le remboursement.")
        st.write(f"✅ Probabilité d’être fiable : **{score_fiable:.2%}**")
        st.write(f"❌ Probabilité d’être risqué : **{score_risque:.2%}**")
        st.progress(score_fiable)
    else:
        st.error("🔴 Le client est considéré comme **risqué** pour le remboursement.")
        st.write(f"✅ Probabilité d’être fiable : **{score_fiable:.2%}**")
        st.write(f"❌ Probabilité d’être risqué : **{score_risque:.2%}**")
        st.progress(score_fiable)  # toujours basé sur la classe 0
# --- SHAP EXPLICATION ---
#Créer un explainer SHAP (à faire 1 seule fois au début de l'app si tu veux optimiser)
    explainer = shap.TreeExplainer(model)
    
#Calcul des valeurs SHAP pour la prédiction de l'utilisateur
    shap_values = explainer.shap_values(input_df)

# Afficher le top des features influentes avec bar plot
    st.subheader("📊 Explication de la prédiction")
    st.write("Voici les variables qui ont le plus influencé la décision du modèle pour ce client :")

# Affichage graphique SHAP dans Streamlit
    plt.figure(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0], max_display=10)
    st.pyplot(plt.gcf())
    st.subheader("📊 Explication de la prédiction")

    st.markdown("""🧠 *Les variables ci-dessous incluent aussi des colonnes internes non visibles dans le formulaire, mais qui influencent la prédiction (valeurs moyennes par défaut).*""")
  


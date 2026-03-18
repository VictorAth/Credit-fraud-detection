import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configuration de la page
st.set_page_config(page_title="Fraud Sentinel AI", layout="wide", page_icon="🛡️")

# 2. Chargement des ressources (Modèle + Scaler)
@st.cache_resource
def load_assets():
    model = joblib.load('model_fraude.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement des fichiers .pkl : {e}")
    st.stop()

# --- INTERFACE ---
st.title("🛡️ Fraud Sentinel : Station d'Analyse IA")
st.markdown("---")

# Utilisation de colonnes pour une interface propre
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📥 Paramètres de la Transaction")
    
    # Sliders basés sur tes analyses de corrélation
    v14 = st.slider("Indicateur V14 (Très critique)", -20.0, 10.0, 0.0, help="Forte corrélation négative")
    v12 = st.slider("Indicateur V12", -20.0, 10.0, 0.0)
    v10 = st.slider("Indicateur V10", -25.0, 10.0, 0.0)
    v11 = st.slider("Indicateur V11 (Positif)", -5.0, 15.0, 0.0, help="Une valeur haute augmente le risque")
    
    amount = st.number_input("Montant de la transaction (€)", min_value=0.0, value=100.0)
    
    # Le temps est mis à 0 par défaut pour la simulation
    time_sim = 0.0
    
    analyze_btn = st.button("🚀 Lancer l'Analyse Score-IA", use_container_width=True)

with col2:
    st.subheader("📊 Résultat du Diagnostic")
    
    if analyze_btn:
        # --- PRÉPARATION DES DONNÉES ---
        # Définition de l'ordre exact des colonnes attendu par ton modèle
        columns = ['scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
                    'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
        
        # Création d'un DataFrame d'une ligne rempli de zéros
        inputs = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        # Injection des variables utilisateur
        inputs['V14'] = v14
        inputs['V12'] = v12
        inputs['V10'] = v10
        inputs['V11'] = v11
        
        # Application du RobustScaler sur le montant (CRUCIAL)
        # On simule aussi le scaling du temps à 0 pour la cohérence
        inputs['scaled_amount'] = scaler.transform([[amount]])[0][0]
        inputs['scaled_time'] = 0.0 

        # --- PRÉDICTION ---
        proba = model.predict_proba(inputs)[0] 
        risk_score = proba[1] * 100
        
        # Affichage du score
        st.write(f"**Score de Risque : {risk_score:.1f}%**")
        
        if risk_score < 30:
            st.progress(risk_score / 100)
            st.success("✅ TRANSACTION LÉGITIME")
        elif risk_score < 75:
            st.progress(risk_score / 100)
            st.warning("⚠️ TRANSACTION SUSPECTE")
        else:
            st.progress(risk_score / 100)
            st.error("🚨 ALERTE : FRAUDE PROBABLE")

        # --- RECOMMANDATIONS ---
        st.markdown("### 📝 Recommandations Opérationnelles")
        if risk_score < 20:
            st.info("**Action :** Validation immédiate.")
        elif 20 <= risk_score < 50:
            st.info("**Action :** Demander une vérification 3D Secure.")
        elif 50 <= risk_score < 80:
            st.warning("**Action :** Mise en attente. Appel du porteur nécessaire.")
        else:
            st.error("**Action :** BLOCAGE IMMÉDIAT. Signalement Conformité.")

        # --- EXPLICATION (XAI) ---
        with st.expander("Pourquoi ce score ?"):
            st.write("Analyse des facteurs dominants :")
            if v14 < -5: st.write("- 🚩 **V14 Bas :** Fort indicateur historique de fraude.")
            if v11 > 5: st.write("- 🚩 **V11 Haut :** Pattern de comportement anormal détecté.")
            if amount > 1000: st.write("- 🚩 **Montant :** Supérieur à la médiane de l'échantillon.")
    else:
        st.info("Ajustez les curseurs et lancez l'analyse.")
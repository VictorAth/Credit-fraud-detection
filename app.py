import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configuration de la page
st.set_page_config(page_title="Fraud Sentinel AI", layout="wide", page_icon="🛡️")

# 2. Chargement des ressources (Modèle + Scaler)
@st.cache_resource
def load_assets():
    # Assure-toi que ces fichiers sont bien à la racine de ton GitHub
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

# --- NOUVEAU : GUIDE D'UTILISATION ET INTERPRÉTATION ---
with st.expander("📖 Guide d'utilisation et Aide à l'interprétation"):
    st.markdown("""
    ### Bienvenue dans Fraud Sentinel
    Cette application utilise un modèle de **Régression Logistique** pour prédire si une transaction est frauduleuse.
    
    **Comment ça marche ?**
    Les indicateurs **V10, V11, V12 et V14** sont des variables issues d'une Analyse en Composantes Principales (PCA). Elles résument des comportements complexes (lieu, type de terminal, fréquence d'achat) tout en protégeant l'anonymat des clients.
    
    **Interprétation du Score :**
    * 🟢 **0% - 30% (Légitime) :** Aucun signal suspect détecté.
    * 🟡 **31% - 75% (Suspect) :** Pattern inhabituel. Une double authentification (3D Secure) est conseillée.
    * 🔴 **76% - 100% (Fraude) :** Très forte ressemblance avec les schémas de fraude répertoriés. Blocage recommandé.
    """)

# Utilisation de colonnes pour une interface propre
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📥 Paramètres de la Transaction")
    st.info("Ajustez les curseurs pour simuler une transaction.")
    
    # Sliders avec explications simplifiées (le "Portrait-Robot")
    v14 = st.slider("Indicateur V14 (Comportement Géographique)", -20.0, 10.0, 0.0, 
                    help="Plus cette valeur est basse, plus l'achat s'éloigne des habitudes du client.")
    
    v12 = st.slider("Indicateur V12 (Fiabilité du Terminal)", -20.0, 10.0, 0.0,
                    help="Analyse si l'appareil ou le site utilisé est habituel ou suspect.")
    
    v10 = st.slider("Indicateur V10 (Fréquence/Rythme d'achat)", -25.0, 10.0, 0.0,
                    help="Détermine si la vitesse des transactions est inhabituelle.")
    
    v11 = st.slider("Indicateur V11 (Niveau d'Anomalie Globale)", -5.0, 15.0, 0.0,
                    help="Une valeur élevée renforce les autres signaux d'alerte.")
    
    amount = st.number_input("Montant de la transaction (€)", min_value=0.0, value=100.0)
    
    analyze_btn = st.button("🚀 Lancer l'Analyse Score-IA", use_container_width=True)

with col2:
    st.subheader("📊 Résultat du Diagnostic")
    
    if analyze_btn:
        # --- PRÉPARATION DES DONNÉES ---
        # Ordre exact des colonnes utilisé pendant l'entraînement (Phase 4)
        columns = ['scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
                   'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
        
        inputs = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        # Injection des variables
        inputs['V14'] = v14
        inputs['V12'] = v12
        inputs['V10'] = v10
        inputs['V11'] = v11
        
        # Mise à l'échelle du montant avec le RobustScaler
        inputs['scaled_amount'] = scaler.transform([[amount]])[0][0]
        inputs['scaled_time'] = 0.0 # Simulation d'un temps neutre

        # --- PRÉDICTION ---
        proba = model.predict_proba(inputs)[0] 
        risk_score = proba[1] * 100
        
        # Affichage visuel du score
        st.write(f"**Score de Risque calculé par l'IA : {risk_score:.1f}%**")
        
        if risk_score < 30:
            st.progress(risk_score / 100)
            st.success("✅ TRANSACTION LÉGITIME")
        elif risk_score < 75:
            st.progress(risk_score / 100)
            st.warning("⚠️ TRANSACTION SUSPECTE")
        else:
            st.progress(risk_score / 100)
            st.error("🚨 ALERTE : FRAUDE PROBABLE")

        # --- RECOMMANDATIONS OPÉRATIONNELLES ---
        st.markdown("### 📝 Recommandations pour l'Agent")
        if risk_score < 20:
            st.info("**Action :** Validation immédiate. Risque négligeable.")
        elif 20 <= risk_score < 50:
            st.info("**Action :** Demander une vérification 3D Secure (SMS).")
        elif 50 <= risk_score < 80:
            st.warning("**Action :** Mise en attente. Appel de confirmation nécessaire.")
        else:
            st.error("**Action :** BLOCAGE IMMÉDIAT. Signalement au département conformité.")

        # --- EXPLICATION LITE (XAI) ---
        with st.expander("🔍 Pourquoi ce score ? (Détails techniques)"):
            st.write("Analyse des facteurs de risque :")
            if v14 < -5: st.write("- 🚩 **V14 Bas :** L'écart géographique ou comportemental est très élevé.")
            if v11 > 5: st.write("- 🚩 **V11 Haut :** Une anomalie structurelle a été détectée dans la transaction.")
            if amount > 2000: st.write("- 🚩 **Montant :** La valeur dépasse les seuils de vigilance standard.")
    else:
        st.info("Ajustez les paramètres à gauche et cliquez sur le bouton pour analyser.")
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Fraud Sentinel AI",
    layout="wide",
    page_icon="🛡️"
)

# 2. CHARGEMENT DES RESSOURCES (Modèle + Scaler)
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

# --- ENTÊTE ---
st.title("🛡️ Fraud Sentinel : Station d'Analyse IA")
st.write("Outil d'aide à la décision pour analystes fraude et gestionnaires e-commerce.")
st.markdown("---")

# --- GUIDE D'UTILISATION ---
with st.expander("❓ C'est quoi cette application et comment l'utiliser ?"):
    st.markdown("""
    ### 🧠 Le Concept
    Cette IA agit comme un **vigile virtuel**. Elle compare les détails d'une transaction avec des milliers de fraudes passées pour calculer un score de ressemblance.
    
    ### 📖 Guide Rapide
    1. **Ajustez les curseurs** à gauche selon les détails reçus dans l'alerte.
    2. **Plus un curseur est à gauche**, plus l'indicateur est jugé "anormal" par rapport aux habitudes du client.
    3. **Observez le diagnostic** : le code couleur (Vert, Orange, Rouge) vous indique l'action à prendre immédiatement.
    """)

# --- MISE EN PAGE : DEUX COLONNES ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📥 Détails de la transaction")
    st.info("Simulez les paramètres de la transaction suspecte ci-dessous.")
    
    # Sliders avec noms explicites pour l'utilisateur
    v14 = st.slider(
        "📍 Localisation (Habituelle vs Inconnue)", 
        -20.0, 10.0, 0.0, 
        help="V14 : Plus la valeur est basse, plus le lieu d'achat est inhabituel pour ce client."
    )
    
    v12 = st.slider(
        "💻 Sécurité du site/appareil", 
        -20.0, 10.0, 0.0,
        help="V12 : Analyse si le terminal ou le site web présente des signes de piratage."
    )
    
    v10 = st.slider(
        "⏳ Rythme des achats", 
        -25.0, 10.0, 0.0,
        help="V10 : Détermine si la vitesse des transactions ressemble à un comportement de robot."
    )
    
    v11 = st.slider(
        "⚠️ Signal d'alerte technique", 
        -5.0, 15.0, 0.0,
        help="V11 : Mesure le niveau global d'anomalies techniques durant le paiement."
    )
    
    amount = st.number_input("Montant de la transaction (€)", min_value=0.0, value=100.0)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Lancer le diagnostic IA", use_container_width=True)

with col2:
    st.subheader("📊 Résultat du Diagnostic")
    
    if analyze_btn:
        # --- PRÉPARATION DES DONNÉES ---
        columns = [
            'scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
            'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
        ]
        
        # Création du DataFrame avec toutes les colonnes à zéro
        inputs = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        # Injection des valeurs des sliders
        inputs['V14'] = v14
        inputs['V12'] = v12
        inputs['V10'] = v10
        inputs['V11'] = v11
        
        # Application du scaler sur le montant
        inputs['scaled_amount'] = scaler.transform([[amount]])[0][0]
        inputs['scaled_time'] = 0.0 

        # --- PRÉDICTION ---
        proba = model.predict_proba(inputs)[0] 
        risk_score = proba[1] * 100
        
        # --- AFFICHAGE DU SCORE ---
        st.write(f"### Score de Risque : **{risk_score:.1f}%**")
        
        if risk_score < 30:
            st.progress(risk_score / 100)
            st.success("✅ **TRANSACTION LÉGITIME** : Risque faible, validation recommandée.")
        elif risk_score < 75:
            st.progress(risk_score / 100)
            st.warning("⚠️ **TRANSACTION SUSPECTE** : Des patterns inhabituels ont été détectés.")
        else:
            st.progress(risk_score / 100)
            st.error("🚨 **ALERTE : FRAUDE PROBABLE** : Très forte ressemblance avec des vols connus.")

        # --- RECOMMANDATIONS OPÉRATIONNELLES ---
        st.markdown("#### 📝 Recommandations pour l'analyste")
        if risk_score < 20:
            st.info("**Action :** Validation automatique immédiate.")
        elif 20 <= risk_score < 50:
            st.info("**Action :** Déclencher une authentification forte (3D Secure / SMS).")
        elif 50 <= risk_score < 80:
            st.warning("**Action :** Mise en attente. Appel téléphonique du client nécessaire.")
        else:
            st.error("**Action :** BLOCAGE IMMÉDIAT. Procédure de signalement conformité.")

        # --- EXPLICATION (XAI) ---
        with st.expander("🔍 Pourquoi ce score ? (Détails techniques)"):
            st.write("Facteurs influençant ce résultat :")
            if v14 < -5: st.write("- 🚩 **Localisation** : L'écart avec les habitudes est significatif.")
            if v11 > 5: st.write("- 🚩 **Technique** : Trop d'anomalies systèmes détectées.")
            if amount > 2000: st.write("- 🚩 **Montant** : Transaction supérieure au seuil de vigilance.")
    else:
        st.info("Modifiez les paramètres à gauche et cliquez sur le bouton pour obtenir une analyse.")
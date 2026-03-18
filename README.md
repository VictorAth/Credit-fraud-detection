# 🛡️ Fraud Sentinel AI : Système de Détection de Fraude Bancaire

Ce projet est une solution de Machine Learning "End-to-End" permettant d'identifier les transactions bancaires frauduleuses en temps réel. Développé dans le cadre de ma licence en **IA et Big Data (IA-BD)**.

## 🚀 Application Live
Accédez à l'interface interactive ici : 
👉 [https://credit-fraud-detection-victorath.streamlit.app/](https://credit-fraud-detection-victorath.streamlit.app/)

## 📊 Présentation du Projet
Le défi principal de ce projet était de traiter un dataset hautement déséquilibré (seulement 0,17% de fraudes). 

**Points techniques clés :**
- **Prétraitement :** Normalisation avec `RobustScaler` et élimination des valeurs aberrantes (Outliers) via la méthode IQR.
- **Équilibrage :** Utilisation du sous-échantillonnage (Under-sampling) pour entraîner le modèle sur des patterns de fraude clairs.
- **Algorithme :** Régression Logistique optimisée pour le **Recall**.

### Performances du Modèle
* **AUC Score :** 0.98 (Excellente capacité de séparation)
* **Recall :** 91% (Priorité absolue à la détection des fraudes pour minimiser les pertes bancaires)
* **F1-Score :** 0.92

## 🧑‍💼 Rôle de l'Analyste Fraude
Cette application n'est pas qu'un simple algorithme ; c'est un **outil d'aide à la décision** destiné aux analystes en milieu bancaire.

**Comment l'analyste utilise l'outil :**
1. **Réception de l'alerte :** Le système central de la banque signale une transaction suspecte avec des indices techniques (V10, V12, V14).
2. **Simulation :** L'analyste entre ces indices dans **Fraud Sentinel** pour obtenir une deuxième opinion de l'IA.
3. **Interprétation :** Grâce aux explications intégrées (XAI), l'analyste comprend *pourquoi* le risque est élevé (ex: anomalie de localisation ou fréquence d'achat suspecte).
4. **Action :** L'analyste décide alors de valider, de demander une vérification SMS ou de bloquer définitivement la carte.

## 🛠️ Stack Technique
* **Langages :** Python 3.10
* **Data Science :** Scikit-Learn, Pandas, NumPy
* **Visualisation :** Matplotlib, Seaborn
* **Déploiement :** Streamlit Cloud, GitHub
* **Environnement :** VS Code, Git Bash

## 📁 Structure du Dépôt
- `app.py` : Code de l'application web.
- `model_fraude.pkl` : Modèle de prédiction sauvegardé.
- `scaler.pkl` : Transformateur pour la mise à l'échelle des données.
- `requirements.txt` : Dépendances pour le serveur de déploiement.

---
*Développé par **Victor Attoh** - Étudiant en IA & Big Data.*

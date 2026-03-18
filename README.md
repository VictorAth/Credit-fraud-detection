# 🛡️ Fraud Sentinel AI : Détection de Fraude Bancaire

Ce projet utilise le Machine Learning pour identifier les transactions bancaires frauduleuses en temps réel. Développé dans le cadre de mon parcours en IA et Big Data.

## 🚀 Application Live
[Lien vers ton application Streamlit ici]

## 📊 Aperçu du Projet
L'objectif est de traiter un dataset hautement déséquilibré (0,17% de fraudes) en utilisant des techniques de sous-échantillonnage (Under-sampling) et de nettoyage d'outliers (méthode IQR).

### Performance du Modèle (Régression Logistique)
* **Recall :** 91% (Priorité à la détection des fraudes)
* **AUC Score :** 0.98
* **F1-Score :** 0.92

## 🛠️ Stack Technique
* **Langage :** Python
* **Bibliothèques :** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Déploiement :** Streamlit Cloud
* **Environnement :** Google Colab / VS Code

## 📁 Structure du Dépôt
* `app.py` : Code de l'interface interactive Streamlit.
* `model_fraude.pkl` : Modèle de Régression Logistique entraîné.
* `scaler.pkl` : RobustScaler utilisé pour la normalisation des montants.
* `requirements.txt` : Liste des dépendances Python.

# Machine Learning – RetentionAI

## 🎯 Objectif

Construire un **modèle de Machine Learning supervisé** capable de prédire le risque de départ (attrition) d’un employé à partir de données RH structurées.

Ce modèle est ensuite exposé via une API FastAPI et utilisé en production.

---

## 📁 Structure du module

```
machine_learning_RetentionAI/
├── app/
│   ├── eda.ipynb
│   └── rf_attrition.pkl
│
├── requirements.txt
└── README.md
```

---

## 📊 Données & Problématique

* Données RH internes (âge, salaire, rôle, satisfaction, ancienneté, etc.)
* Variable cible : **Attrition** (0 = reste, 1 = quitte)
* Problème : **déséquilibre des classes** (peu de départs)

---

## 🔍 Analyse Exploratoire (EDA)

Notebook : `eda.ipynb`

### Étapes réalisées

* Suppression des variables inutiles
* Analyse de la distribution de la cible
* Visualisations avec **Seaborn**
* Étude de corrélation avec la cible
* Analyse des relations métier (ancienneté, salaire, satisfaction)

---

## ⚙️ Préprocessing

* Transformation de la cible (binaire)
* Encodage des variables catégorielles → `OneHotEncoder`
* Normalisation / standardisation
* Séparation : `train_test_split`

---

## 🤖 Modèles testés

* Régression Logistique
* Support Vector Classifier (SVC)
* Random Forest Classifier

### Déséquilibre des classes

* Application de **SMOTE** pour améliorer le rappel sur la classe minoritaire

---

## 📈 Évaluation

Métriques utilisées :

* Matrice de confusion
* Recall (classe Attrition = 1)
* F1-score
* Courbe ROC
* Rapport de classification

---

## 🏆 Résultats comparatifs

| Modèle                | Avant SMOTE             | Après SMOTE                             |
| --------------------- | ----------------------- | --------------------------------------- |
| SVC                   | Recall: 0.50 – F1: 0.46 | Recall_macro: 0.69 – F1_macro: 0.63     |
| Régression Logistique | Recall: 0.55 – F1: 0.47 | Recall_macro: 0.65 – F1_macro: 0.54     |
| **Random Forest**     | Recall: 0.39 – F1: 0.43 | **Recall_macro: 0.67 – F1_macro: 0.70** |

---

## ✅ Pourquoi Random Forest ?

* Meilleur compromis **précision / rappel** après SMOTE
* Bonne capacité à gérer les relations non linéaires
* Robuste au bruit
* Importance des variables interprétable
* Adapté à un contexte métier RH

➡️ **Choisi comme modèle final**

---

## 💾 Sauvegarde du modèle

* Modèle final sauvegardé avec `joblib`
* Fichier : `rf_attrition.pkl`
* Chargé dynamiquement dans l’API backend

---

## 🔌 Intégration Backend

* Chargement via `model_loader.py`
* Endpoint `/predict`
* Retour :

  * `prediction` (0 / 1)
  * `probability`

---

## 🚀 Améliorations futures

* Feature importance SHAP
* Monitoring du modèle (MLflow)
* Réentraînement automatique
* Détection de dérive des données

---

## 🧠 Conclusion

Ce module ML constitue le **cœur décisionnel** de RetentionAI et fournit une prédiction fiable, explicable et exploitable par les équipes RH.

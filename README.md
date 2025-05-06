
# 🔌 Projet de Prédiction de la Consommation d'Électricité

Ce projet a pour objectif de prédire la consommation d'électricité à partir de données historiques, en utilisant des algorithmes d'apprentissage automatique (KNN et Random Forest).

---

## 📁 Structure du projet

```
mspr/
├── data/
│   ├── raw/               # Données brutes (CSV, XLS)
│   └── processed/         # Données nettoyées et encodées
│
├── notebooks/             # Analyses exploratoires et modèles initiaux
├── src/                   # Code source 
├── models/                # Modèles entraînés (.pkl)
├── outputs/
│      # Graphiques générés
│      # Comparaisons de performances
│
├── main.py                # Script principal pour exécuter le pipeline complet
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation du projet
```

---

## ⚙️ Étapes du pipeline

1. **Exploration des données** (`explore_data_modulaire.py`)
   - Affiche un aperçu des données
   - Génère des boxplots pour visualiser la consommation

2. **Encodage** (`encoding_data_modulaire.py`)
   - Encodage One-Hot des variables catégorielles
   - Sauvegarde du fichier encodé

3. **Modèle KNN** (`model1_KNN_modulaire.py`)
   - Entraînement et évaluation d'un modèle K-Nearest Neighbors

4. **Modèle Random Forest** (`model2_Randomforest_modulaire.py`)
   - Entraînement et évaluation d'un modèle Random Forest

---

## ▶️ Lancer le pipeline

Assurez-vous que vos fichiers de données sont présents dans `data/raw/`, puis exécutez :

```bash
python main.py
```

---

## 🧪 Dépendances

Installez les packages nécessaires avec :

```bash
pip install -r requirements.txt
```

---

## 📊 Résultats

Les performances des modèles sont sauvegardées et comparées automatiquement. Les figures générées sont disponibles dans `outputs/figures/`.

---

## 🧠 Modèles utilisés

- `KNeighborsRegressor`
- `RandomForestRegressor`

---

## ✍️ Auteurs

Projet réalisé dans le cadre du MSPR pour prédire la consommation énergétique.

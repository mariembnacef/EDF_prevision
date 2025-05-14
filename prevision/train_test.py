#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'entraînement et d'évaluation de modèle pour prédire la consommation électrique
Ce module contient les fonctions de préparation des données, d'entraînement du modèle XGBoost,
d'évaluation des performances et de visualisation des résultats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
import joblib
from dateutil.easter import easter
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def preparer_donnees(chemin_fichier):
    """
    Fonction pour charger et préparer les données temporelles
    
    Args:
        chemin_fichier (str): Chemin du fichier CSV contenant les données
        
    Returns:
        tuple: (X, y, df_model) - Features, target et dataframe complet
    """
    # Chargement des données
    df = pd.read_csv(chemin_fichier, sep="\t", encoding="latin1")
    
    # Conversion des colonnes Date et Heures
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time
    
    # Variables temporelles
    df['mois'] = df['Date'].dt.month
    df['annee'] = df['Date'].dt.year
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    
    # Encodage des variables catégorielles
    saison_mapping = {'Automne': 1, 'Hiver': 2, 'Printemps': 3, 'Été': 4}
    df['saison_num'] = df['Saison'].map(saison_mapping)
    
    # Vérification de la présence de la colonne 'Type de jour TEMPO'
    if 'Type de jour TEMPO' in df.columns:
        tempo_mapping = {'BLEU': 1, 'BLANC': 2, 'ROUGE': 3}
        df['tempo_num'] = df['Type de jour TEMPO'].map(tempo_mapping)
    else:
        print("⚠️ La colonne 'Type de jour TEMPO' est absente. Cette fonctionnalité ne sera pas utilisée.")
    
    # Encodage période de la journée
    def encoder_periode_jour(time_obj):
        if pd.isnull(time_obj): return None
        if dt.time(0, 0) <= time_obj < dt.time(5, 0): return 1  # nuit
        elif dt.time(5, 0) <= time_obj < dt.time(12, 0): return 2  # matin
        elif dt.time(12, 0) <= time_obj < dt.time(18, 0): return 3  # après-midi
        else: return 4  # soir
    
    df['periode_jour_code'] = df['Heures'].apply(encoder_periode_jour)
    
    # Jours fériés
    def get_french_holidays(year):
        fixed = [
            dt.date(year, 1, 1), dt.date(year, 5, 1), dt.date(year, 5, 8),
            dt.date(year, 7, 14), dt.date(year, 8, 15), dt.date(year, 11, 1),
            dt.date(year, 11, 11), dt.date(year, 12, 25)
        ]
        easter_date = easter(year)
        movable = [
            easter_date + dt.timedelta(days=1),   # Lundi de Pâques
            easter_date + dt.timedelta(days=39),  # Ascension
            easter_date + dt.timedelta(days=50),  # Lundi de Pentecôte
        ]
        return fixed + movable
    
    years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
    all_holidays = [date for year in years for date in get_french_holidays(year)]
    all_holidays = pd.to_datetime(all_holidays)
    
    df['jour_ferie'] = df['Date'].dt.normalize().isin(all_holidays)
    
    # Conversion des heures en format numérique
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M:%S', errors='coerce')
    df['Heures_float'] = df['Heures'].dt.hour + df['Heures'].dt.minute / 60
    
    # Tri chronologique et création de lags
    df = df.sort_values(['Date', 'Heures_float']).reset_index(drop=True)
    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}'] = df['Consommation'].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    
    # Supprimer les colonnes inutiles pour le modèle
    cols_to_drop = ['Type de jour TEMPO', 'Date', 'Heures', 'Prévision J', 
                   'Prévision J-1', 'Jour', 'Saison']
    
    # Ne supprimer que les colonnes qui existent
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df_model = df.drop(columns=cols_to_drop)
    
    df_model = df_model.dropna()
    
    # Séparation features / target
    X = df_model.drop(columns=['Consommation'])
    y = df_model['Consommation']
    
    print(f"Colonnes du modèle: {X.columns.tolist()}")
    
    return X, y, df_model

def entrainer_modele(X, y, verbose=True):
    """
    Fonction pour entraîner et optimiser le modèle XGBoost
    
    Args:
        X (DataFrame): Features
        y (Series): Target
        verbose (bool): Afficher les détails du processus d'entraînement
        
    Returns:
        tuple: (best_model, X_train, X_val, y_train, y_val)
    """
    # Split temporel (pas de shuffle car série temporelle)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Grille large pour exploration RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 300, 500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5],
    }
    
    # Modèle de base
    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Recherche aléatoire des hyperparamètres
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1 if verbose else 0,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    if verbose:
        print("🔍 Meilleurs paramètres RandomizedSearchCV :")
        print(random_search.best_params_)
    
    # Récupérer les meilleurs paramètres comme base
    best_params = random_search.best_params_
    
    # Grille affinée autour des meilleures valeurs
    param_grid = {
        'n_estimators': [max(100, best_params['n_estimators'] - 100), best_params['n_estimators'], best_params['n_estimators'] + 100],
        'learning_rate': [best_params['learning_rate'] / 2, best_params['learning_rate'], best_params['learning_rate'] * 2],
        'max_depth': [max(1, best_params['max_depth'] - 1), best_params['max_depth'], best_params['max_depth'] + 1],
        'subsample': [best_params['subsample']],
        'colsample_bytree': [best_params['colsample_bytree']],
        'reg_alpha': [best_params['reg_alpha']],
        'reg_lambda': [best_params['reg_lambda']]
    }
    
    xgb_tuned = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Recherche sur grille ciblée
    grid_search = GridSearchCV(
        estimator=xgb_tuned,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1 if verbose else 0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print("🏆 Meilleurs paramètres après GridSearch ciblé :")
        print(grid_search.best_params_)
    
    # Récupérer le meilleur modèle
    best_model = grid_search.best_estimator_
    
    return best_model, X_train, X_val, y_train, y_val

def evaluer_modele(model, X_train, X_val, y_train, y_val):
    """
    Fonction pour évaluer les performances du modèle et détecter l'overfitting
    
    Args:
        model: Modèle entraîné
        X_train (DataFrame): Features d'entraînement
        X_val (DataFrame): Features de validation
        y_train (Series): Target d'entraînement
        y_val (Series): Target de validation
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Métriques d'évaluation
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)
    
    # Vérification de l'overfitting
    diff_r2 = r2_train - r2_val
    status = "Overfitting détecté" if diff_r2 > 0.05 else "Pas (ou peu) d'overfitting"
    
    # Affichage des résultats
    print(f"Train R²: {r2_train:.4f}, Train RMSE: {rmse_train:.2f}")
    print(f"Test  R²: {r2_val:.4f}, Test  RMSE: {rmse_val:.2f}")
    print(f"Différence R² (Train - Test): {diff_r2:.4f} → {status}")
    print(f"📉 RMSE finale : {rmse_val:.2f}")
    print(f"📈 R² (score de détermination) : {r2_val:.4f}")
    
    return {
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'rmse_val': rmse_val,
        'r2_val': r2_val,
        'overfitting': diff_r2 > 0.05
    }

def sauvegarder_modele(model, nom_fichier="models/xgboost_conso_best_model.pkl"):
    """
    Fonction pour sauvegarder le modèle entrainé
    
    Args:
        model: Modèle à sauvegarder
        nom_fichier (str): Chemin où sauvegarder le modèle
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(nom_fichier), exist_ok=True)
    
    # Sauvegarder le modèle
    joblib.dump(model, nom_fichier)
    print(f"✅ Modèle sauvegardé sous '{nom_fichier}'")
    
    return nom_fichier

def visualiser_resultats(y_test, y_pred, features_importance=None):
    """
    Fonction pour visualiser les résultats du modèle
    
    Args:
        y_test (Series): Valeurs réelles
        y_pred (array): Prédictions du modèle
        features_importance (Series, optional): Importance des variables
    """
    # Création de la figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    # Premier graphique: prédictions vs réalité
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel('Valeurs réelles')
    axes[0].set_ylabel('Prédictions')
    axes[0].set_title('Comparaison des prédictions avec les valeurs réelles')
    
    # Second graphique: distribution des erreurs
    erreurs = y_test - y_pred
    sns.histplot(erreurs, kde=True, ax=axes[1])
    axes[1].set_xlabel('Erreur de prédiction')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des erreurs de prédiction')
    
    plt.tight_layout()
    plt.show()
    
    # Affichage de l'importance des variables si disponible
    if features_importance is not None:
        plt.figure(figsize=(12, 6))
        features_importance.plot(kind='barh')
        plt.title('Importance des variables')
        plt.tight_layout()
        plt.show()

# Si le script est exécuté directement
if __name__ == "__main__":
    print("Ce module contient des fonctions pour entraîner et évaluer un modèle XGBoost.")
    print("Pour utiliser la pipeline complète, exécutez main.py")
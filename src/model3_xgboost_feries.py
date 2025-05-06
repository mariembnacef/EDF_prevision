
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.record_performance import enregistrer_performances_csv
from datetime import datetime
import holidays
import joblib

def charger_donnees(path='data/processed/df_filtre.csv'):
    df = pd.read_csv(path, sep='\t', encoding='latin1')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Annee'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Jour_semaine'] = df['Date'].dt.dayofweek  # 0 = Lundi
    df['is_holiday'] = df['Date'].isin(holidays.FR(years=range(df['Annee'].min(), df['Annee'].max() + 1)).keys()).astype(int)

    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce')
    df['Heure'] = df['Heures'].dt.hour + df['Heures'].dt.minute / 60.0

    df['Delta_Prev'] = df['Prévision J'] - df['Prévision J-1']
    df.drop(columns=['Heures', 'Date'], inplace=True, errors='ignore')

    # Encodage OneHot
    for col in ['Type de jour TEMPO', 'Jour', 'Saison']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col])

    df.drop(columns=['Heures'], inplace=True, errors='ignore')
    df = df.dropna()
    return df

def preparation_donnees(df):
    X = df.drop(columns=['Consommation'])
    y = df['Consommation']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrainer_modele_xgb_gridsearch(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    return mae, rmse, r2

def sauvegarder_modele(model, path='models/modele_xgboost_feries.pkl'):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def pipeline_xgb_feries():
    print("🔍 Chargement des données avec jours fériés...")
    df = charger_donnees()
    X_train, X_test, y_train, y_test = preparation_donnees(df)

    print("🚀 Entraînement XGBoost avec GridSearchCV...")
    model = entrainer_modele_xgb_gridsearch(X_train, y_train)

    mae, rmse, r2 = evaluer_modele(model, X_test, y_test)
    enregistrer_performances_csv("XGBoost_Feries", mae, rmse, r2)
    model = entrainer_modele_xgb_gridsearch(X_train, y_train)
    sauvegarder_modele(model)
    print("✅ Modèle XGBoost optimisé avec jours fériés entraîné et évalué.")

if __name__ == "__main__":
    pipeline_xgb_feries()

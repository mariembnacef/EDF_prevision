
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.record_performance import enregistrer_performances_csv
import joblib
def charger_donnees(path='data/processed/df_filtre.csv'):
    df = pd.read_csv(path, sep='\t', encoding='latin1')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Annee'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Jour_semaine'] = df['Date'].dt.dayofweek  # 0=Lundi, 6=Dimanche

    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce')
    df['Heure'] = df['Heures'].dt.hour + df['Heures'].dt.minute / 60.0

    df['Delta_Prev'] = df['Prévision J'] - df['Prévision J-1']

    # Encodage one-hot
    for col in ['Type de jour TEMPO', 'Jour', 'Saison']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col])

    df.drop(columns=['Date', 'Heures'], inplace=True, errors='ignore')
    return df

def preparation_donnees(df):
    df = df.dropna()
    X = df.drop(columns=['Consommation'])
    y = df['Consommation']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrainer_modele_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    return mae, rmse, r2
def sauvegarder_modele(model, path='models/modele_xgboost.pkl'):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")
    
def pipeline_xgboost():
    print("🚀 Entraînement du modèle XGBoost avec feature engineering...")
    df = charger_donnees()
    X_train, X_test, y_train, y_test = preparation_donnees(df)
    model = entrainer_modele_xgb(X_train, y_train)
    mae, rmse, r2 = evaluer_modele(model, X_test, y_test)
    enregistrer_performances_csv("XGBoost", mae, rmse, r2)
    model = entrainer_modele_xgb(X_train, y_train)
    sauvegarder_modele(model)
    print("✅ Modèle XGBoost enregistré avec ses performances.")

if __name__ == "__main__":
    pipeline_xgboost()

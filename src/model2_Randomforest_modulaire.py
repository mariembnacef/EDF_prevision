
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.record_performance import enregistrer_performances_csv


def charger_donnees(path='data/processed/df_encoded.csv'):
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        print(f"Données chargées depuis {path}")
        return df
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable à {path}")
        return None

def preparation_donnees(df):
    X = df.drop(columns=['Consommation'])
    y = df['Consommation']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrainer_modele_rf(X_train, y_train, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    enregistrer_performances_csv("Random Forest", mae, rmse, r2)

    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    return mae, rmse, r2

def sauvegarder_modele(model, path='models/modele_random_forest.pkl'):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")

def pipeline_rf():
    df = charger_donnees()
    if df is not None:
        X_train, X_test, y_train, y_test = preparation_donnees(df)
        model = entrainer_modele_rf(X_train, y_train)
        evaluer_modele(model, X_test, y_test)
        sauvegarder_modele(model)

if __name__ == "__main__":
    pipeline_rf()

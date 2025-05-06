
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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

def entrainer_modele_knn(X_train, y_train, n_neighbors=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluer_modele(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)
    enregistrer_performances_csv("KNN", mae, rmse, r2)

    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    return mae, rmse, r2

def sauvegarder_modele(model, scaler, model_path='models/modele_knn.pkl', scaler_path='models/scaler_knn.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modèle et scaler sauvegardés dans {model_path} et {scaler_path}")

def pipeline_knn():
    df = charger_donnees()
    if df is not None:
        X_train, X_test, y_train, y_test = preparation_donnees(df)
        model, scaler = entrainer_modele_knn(X_train, y_train)
        evaluer_modele(model, scaler, X_test, y_test)
        sauvegarder_modele(model, scaler)

if __name__ == "__main__":
    pipeline_knn()

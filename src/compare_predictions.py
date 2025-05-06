
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error

def charger_donnees(path='data/processed/df_encoded.csv'):
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df = df.dropna()
    X = df.drop(columns=['Consommation'])
    y = df['Consommation']
    return X, y

def charger_modeles(modeles_info):
    modeles = {}
    for nom, chemins in modeles_info.items():
        modele = joblib.load(chemins['modele'])
        scaler = joblib.load(chemins['scaler']) if 'scaler' in chemins else None
        modeles[nom] = {'modele': modele, 'scaler': scaler}
    return modeles

def comparer_predictions():
    X, y = charger_donnees()
    modeles_info = {
        'KNN': {'modele': 'models/modele_knn.pkl', 'scaler': 'models/scaler_knn.pkl'},
        'Random Forest': {'modele': 'models/modele_random_forest.pkl'},
        'XGBoost': {'modele': 'models/modele_xgboost.pkl'},
        'XGBoost_Feries': {'modele': 'models/modele_xgboost_feries.pkl'}
    }

    modeles = charger_modeles(modeles_info)

    plt.figure(figsize=(14, 6))
    plt.plot(y.values[:200], label='Réel', linewidth=2, color='black')

    for nom, obj in modeles.items():
        X_input = X.copy()
        if obj['scaler']:
            X_input = obj['scaler'].transform(X_input)
        y_pred = obj['modele'].predict(X_input)
        plt.plot(y_pred[:200], label=nom)

    plt.title("Comparaison des modèles - Réel vs Prédictions")
    plt.xlabel("Échantillons")
    plt.ylabel("Consommation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/predictions_vs_reel.png")
    plt.close()
    print("✅ Graphique sauvegardé : outputs/figures/predictions_vs_reel.png")

if __name__ == "__main__":
    comparer_predictions()

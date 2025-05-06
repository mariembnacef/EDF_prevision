
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

def charger_donnees(path='data/processed/df_filtre.csv'):
    """Charge le fichier CSV contenant les données filtrées."""
    try:
        df = pd.read_csv(path, sep='\t', encoding='latin1')
        print(f"Données chargées avec succès depuis {path}")
        print(df.columns)
        return df
    except FileNotFoundError:
        print(f"Erreur : fichier non trouvé à {path}")
        return None

def encoder_variables(df):
    """Ajoute des colonnes temporelles et encode les variables catégorielles."""
    # Conversion de la colonne 'Date' en datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Annee'] = df['Date'].dt.year
        df['Mois'] = df['Date'].dt.month
        df.drop(columns=['Date'], inplace=True)
        print("Colonnes 'Annee' et 'Mois' extraites depuis 'Date'.")
    
    # Encodage de l'heure
    if 'Heures' in df.columns:
        df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce')
        df['Heure'] = df['Heures'].dt.hour + df['Heures'].dt.minute / 60.0
        df.drop(columns=['Heures'], inplace=True)
        print("Colonne 'Heure' extraite depuis 'Heures'.")
    
    # Colonnes à encoder avec OneHot
    colonnes_a_encoder = ['Type de jour TEMPO', 'Jour', 'Saison']
    for col in colonnes_a_encoder:
        if col in df.columns:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(df[[col]])
            cols = encoder.get_feature_names_out([col])
            df_encoded = pd.DataFrame(encoded, columns=cols, index=df.index)
            df = pd.concat([df.drop(columns=[col]), df_encoded], axis=1)
            print(f"Encodage de '{col}' effectué.")
        else:
            print(f"'{col}' non trouvé dans les colonnes.")
    
    return df

def enregistrer_donnees(df, path='data/processed/df_encoded.csv'):
    """Enregistre le DataFrame encodé dans un fichier CSV."""
    df.to_csv(path, sep=';', index=False, encoding='utf-8')
    print(f"Données encodées enregistrées dans {path}")

def pipeline_encodage():
    """Pipeline complet d'encodage."""
    df = charger_donnees()
    if df is not None:
        df = encoder_variables(df)
        enregistrer_donnees(df)

if __name__ == "__main__":
    pipeline_encodage()

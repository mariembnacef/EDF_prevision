import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_df(path):
    df = pd.read_csv(path, sep="\t", encoding="latin1", index_col = False)
    return df

path="../data/raw/annuel"
path_calendar="../data/raw/calendar"

def merge_from_folder(folder_path):
    """
    Utilise read_df pour lire et fusionner tous les fichiers .csv ou .txt d'un dossier.

    Args:
        folder_path (str): Chemin du dossier contenant les fichiers.

    Returns:
        pd.DataFrame: DataFrame fusionné.
    """
    all_files = [f for f in os.listdir(folder_path)]
    dfs = []
    
    for file in all_files:
        full_path = os.path.join(folder_path, file)
        df = read_df(full_path)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

annuel =  merge_from_folder(path)
calendrier =  merge_from_folder(path_calendar)
doublons = annuel[annuel.duplicated()]
print(doublons)

calendrier["Date"] = pd.to_datetime(calendrier["Date"], errors='coerce')

# Supprimer les lignes dont la colonne 'Date' est NaT (donc non valides)
calendrier = calendrier.dropna(subset=["Date"]).reset_index(drop=True)

annuel.replace("ND", pd.NA, inplace=True)
annuel=annuel[annuel["Périmètre"] == "France"]
annuel.replace("ND", np.nan, inplace=True)
missing_counts = annuel.isna().sum()

def fusionner_par_date(annuel: pd.DataFrame, calendrier: pd.DataFrame, type_jointure: str = "inner") -> pd.DataFrame:
    """
    Fusionne deux DataFrames sur la colonne 'Date' après conversion en datetime.

    Args:
        annuel (pd.DataFrame): DataFrame contenant les données principales.
        calendrier (pd.DataFrame): DataFrame contenant les infos de calendrier.
        type_jointure (str): Type de jointure ('inner', 'left', 'right', 'outer').

    Returns:
        pd.DataFrame: DataFrame fusionné sur la colonne 'Date'.
    """
    # Conversion en datetime
    annuel['Date'] = pd.to_datetime(annuel['Date'], errors='coerce')
    calendrier['Date'] = pd.to_datetime(calendrier['Date'], errors='coerce')

    # Suppression des lignes avec dates invalides (optionnel)
    annuel = annuel.dropna(subset=['Date'])
    calendrier = calendrier.dropna(subset=['Date'])

    # Fusion
    fusion = pd.merge(annuel, calendrier, on='Date', how=type_jointure)
    return fusion


df_final = fusionner_par_date(annuel, calendrier, type_jointure="inner")

def garder_colonnes_utiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les colonnes essentielles dans le DataFrame.

    Colonnes conservées :
        - 'Type de jour TEMPO'
        - 'Date'
        - 'Heures'
        - 'Prévision J'
        - 'Prévision J-1'
        - 'Consommation'

    Args:
        df (pd.DataFrame): Le DataFrame d'origine.

    Returns:
        pd.DataFrame: Le DataFrame filtré avec uniquement les colonnes souhaitées.
    """
    colonnes_a_garder = [
        'Type de jour TEMPO',
        'Date',
        'Heures',
        'Prévision J',
        'Prévision J-1',
        'Consommation'
    ]
    
    # On garde uniquement celles qui existent dans le DataFrame
    colonnes_presentes = [col for col in colonnes_a_garder if col in df.columns]
    
    return df[colonnes_presentes].copy()

df_v1 = garder_colonnes_utiles(df_final)
df_nan_conso = df_v1[df_v1['Consommation'].isna()][['Date', 'Heures','Consommation']]
df_nan_conso
df_nan_conso["Date"].unique()

def supprimer_nan_quart_impair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes où 'Heures' est à :15 ou :45 et 'Consommation' est NaN.

    Args:
        df (pd.DataFrame): Le DataFrame d'origine.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    # Extraire les minutes comme entier (ex: 15, 30, 45, ...)
    df['Minutes'] = df['Heures'].astype(str).str.slice(3, 5).astype(int)

    # Identifier les lignes à supprimer : minutes = 15 ou 45 ET consommation = NaN
    condition_suppr = df['Minutes'].isin([15, 45]) & df['Consommation'].isna()

    # Supprimer ces lignes
    df_filtré = df[~condition_suppr].copy()

    # Nettoyage : on peut retirer la colonne temporaire si besoin
    df_filtré.drop(columns='Minutes', inplace=True)

    return df_filtré

df_filtre = supprimer_nan_quart_impair(df_v1)

def comparer_dataframes(df_original: pd.DataFrame, df_nettoye: pd.DataFrame):
    """
    Affiche un tableau comparatif des données avant et après nettoyage.
    """
    total_original = len(df_original)
    total_nettoye = len(df_nettoye)
    nan_original = df_original['Consommation'].isna().sum()
    nan_nettoye = df_nettoye['Consommation'].isna().sum()

    comparatif = pd.DataFrame({
        "Avant nettoyage (df_v1)": [total_original, nan_original, total_original - total_nettoye],
        "Après nettoyage (df_filtré)": [total_nettoye, nan_nettoye, None]
    }, index=["Nombre de lignes", "NaN dans 'Consommation'", "Lignes supprimées"])
    
    return comparatif

df_comparatif = comparer_dataframes(df_v1, df_filtre)

def ajouter_infos_temporelles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes 'Jour', 'Weekend' et 'Saison' à partir de la colonne 'Date'.
    """
    # Assurer que 'Date' est bien au format datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Jour de la semaine - Sans utiliser locale qui cause l'erreur
    # Utiliser une approche alternative pour avoir les jours en français
    jours_semaine = {
        0: 'Lundi',
        1: 'Mardi',
        2: 'Mercredi',
        3: 'Jeudi',
        4: 'Vendredi',
        5: 'Samedi',
        6: 'Dimanche'
    }
    df['Jour'] = df['Date'].dt.weekday.map(jours_semaine)

    # Weekend (Samedi = 5, Dimanche = 6)
    df['Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)

    # Mois pour construire les saisons
    mois = df['Date'].dt.month

    # Fonction d'attribution des saisons
    def determiner_saison(m):
        if m in [12, 1, 2]:
            return 'Hiver'
        elif m in [3, 4, 5]:
            return 'Printemps'
        elif m in [6, 7, 8]:
            return 'Été'
        else:
            return 'Automne'

    df['Saison'] = mois.apply(determiner_saison)

    return df

df_filtre = ajouter_infos_temporelles(df_filtre)
df_filtre.to_csv("../data/df_filtre.csv", sep="\t", index=False, encoding="latin1")
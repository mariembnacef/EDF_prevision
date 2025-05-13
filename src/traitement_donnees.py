
import holidays
import pandas as pd
import os
import numpy as np
from dateutil.easter import easter
import datetime as dt

def read_df(path):
    return pd.read_csv(path, sep="\t", encoding="latin1", index_col=False)


def merge_from_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path)]
    dfs = [read_df(os.path.join(folder_path, file)) for file in all_files]
    return pd.concat(dfs, ignore_index=True)


def nettoyer_calendrier(df):
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df.dropna(subset=["Date"]).reset_index(drop=True)


def nettoyer_annuel(df):
    df.replace("ND", pd.NA, inplace=True)
    df = df[df["Périmètre"] == "France"]
    df.replace("ND", np.nan, inplace=True)
    return df


def fusionner_par_date(annuel, calendrier, type_jointure="inner"):
    annuel['Date'] = pd.to_datetime(annuel['Date'], errors='coerce')
    calendrier['Date'] = pd.to_datetime(calendrier['Date'], errors='coerce')
    annuel = annuel.dropna(subset=['Date'])
    calendrier = calendrier.dropna(subset=['Date'])
    return pd.merge(annuel, calendrier, on='Date', how=type_jointure)


def garder_colonnes_utiles(df):
    colonnes_a_garder = ['Type de jour TEMPO', 'Date', 'Heures', 'Consommation']
    colonnes_presentes = [col for col in colonnes_a_garder if col in df.columns]
    return df[colonnes_presentes].copy()


def supprimer_nan_quart_impair(df):
    df['Minutes'] = df['Heures'].astype(str).str.slice(3, 5).astype(int)
    condition_suppr = df['Minutes'].isin([15, 45]) & df['Consommation'].isna()
    df_filtré = df[~condition_suppr].copy()
    df_filtré.drop(columns='Minutes', inplace=True)
    return df_filtré


# def ajouter_infos_temporelles(df):
    # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # df['Jour'] = df['Date'].dt.day_name(locale='fr_FR')
    # df['Mois'] = df['Date'].dt.month
    # 
    # df['Annee'] = df['Date'].dt.year
# 
    # df['Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
    # mois = df['Date'].dt.month
# 
    # def determiner_saison(m):
        # if m in [12, 1, 2]:
            # return 'Hiver'
        # elif m in [3, 4, 5]:
            # return 'Printemps'
        # elif m in [6, 7, 8]:
            # return 'Été'
        # else:
            # return 'Automne'
# 
    # years = df['Date'].dt.year.unique()
    # jours_feries = holidays.France(years=years)
    # df['JourFerie'] = df['Date'].isin(jours_feries).astype(int)
    # df['Saison'] = mois.apply(determiner_saison)
    # df=df.drop(columns=['Date'])
    # return df
def determiner_saison(m):
    if m in [12, 1, 2]:
       return 'Hiver'
    elif m in [3, 4, 5]:
      return 'Printemps'
    elif m in [6, 7, 8]:
      return 'Été'
    else:
      return 'Automne'
def ajouter_infos_temporelles(df):
    # Conversion des colonnes Date et Heures
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time

    # Mois (1 à 12)
    df['mois'] = df['Date'].dt.month

    # Année
    df['annee'] = df['Date'].dt.year

    # Jour de la semaine (1 = lundi, ..., 7 = dimanche)
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    df['Saison'] = df['mois'].apply(determiner_saison)
    # Saisons encodées (1 = Automne, 2 = Hiver, 3 = Printemps, 4 = Été)
    saison_mapping = {'Automne': 1, 'Hiver': 2, 'Printemps': 3, 'Été': 4}
    df['saison_num'] = df['Saison'].map(saison_mapping)

    # Couleur TEMPO encodée (1 = BLEU, 2 = BLANC, 3 = ROUGE)
    tempo_mapping = {'BLEU': 1, 'BLANC': 2, 'ROUGE': 3}
    df['tempo_num'] = df['Type de jour TEMPO'].map(tempo_mapping)

    # Encodage période de la journée (1 = nuit, 2 = matin, 3 = après-midi, 4 = soir)
    def encoder_periode_jour(time_obj):
        if pd.isnull(time_obj): return None
        if dt.time(0, 0) <= time_obj < dt.time(5, 0): return 1  # nuit
        elif dt.time(5, 0) <= time_obj < dt.time(12, 0): return 2  # matin
        elif dt.time(12, 0) <= time_obj < dt.time(18, 0): return 3  # après-midi
        else: return 4  # soir

    df['periode_jour_code'] = df['Heures'].apply(encoder_periode_jour)

    # Jours fériés (fixes + mobiles)
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
    df=df.drop(columns=['Date'])
    return df

def main():
    path_annuel = "../data/raw/annuel data"
    path_calendar = "../data/raw/calendar data"
    output_path = "../data/processed/df_filtre.csv"

    annuel = nettoyer_annuel(merge_from_folder(path_annuel))
    calendrier = nettoyer_calendrier(merge_from_folder(path_calendar))
    df_final = fusionner_par_date(annuel, calendrier)
    df_v1 = garder_colonnes_utiles(df_final)
    df_filtré = supprimer_nan_quart_impair(df_v1)
    df_enrichi = ajouter_infos_temporelles(df_filtré)

    df_enrichi.to_csv(output_path, sep="\t", index=False, encoding="latin1")
    print(f"Fichier sauvegardé sous : {output_path}")


if __name__ == "__main__":
    main()

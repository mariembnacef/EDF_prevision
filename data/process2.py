import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
import datetime as dt  # Move this import to the top
from dateutil.easter import easter  # Move this import to the top

df = pd.read_csv("df_filtre.csv", sep="\t", encoding="latin1")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Définir les jours fériés fixes
def get_french_fixed_holidays(year):
    return [
        dt.date(year, 1, 1),    # Jour de l'an
        dt.date(year, 5, 1),    # Fête du Travail
        dt.date(year, 5, 8),    # Victoire 1945
        dt.date(year, 7, 14),   # Fête nationale
        dt.date(year, 8, 15),   # Assomption
        dt.date(year, 11, 1),   # Toussaint
        dt.date(year, 11, 11),  # Armistice
        dt.date(year, 12, 25),  # Noël
    ]

# Définir les jours fériés mobiles
def get_french_movable_holidays(year):
    easter_date = easter(year)
    return [
        easter_date + dt.timedelta(days=1),   # Lundi de Pâques
        easter_date + dt.timedelta(days=39),  # Ascension
        easter_date + dt.timedelta(days=50),  # Lundi de Pentecôte
    ]

# Générer la liste de tous les jours fériés pour la période du dataset
years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
all_holidays = []

for year in years:
    all_holidays.extend(get_french_fixed_holidays(year))
    all_holidays.extend(get_french_movable_holidays(year))

# Convertir les jours fériés en datetime pour comparaison
all_holidays = pd.to_datetime(all_holidays)

# Créer la colonne 'jour_ferie'
df['jour_ferie'] = df['Date'].dt.normalize().isin(all_holidays)

# Afficher un échantillon
print(df[['Date', 'jour_ferie']].head())

df['mois'] = df['Date'].dt.month
df['jour_semaine'] = df['Date'].dt.weekday + 1
print(df['Saison'].unique())

# Dictionnaire de correspondance saison → numéro
saison_mapping = {
    'Automne': 1,
    'Hiver': 2,
    'Printemps': 3,
    'Été': 4
}

# Créer la colonne 'saison_num'
df['saison_num'] = df['Saison'].map(saison_mapping)

# Dictionnaire de correspondance TEMPO → numéro
tempo_mapping = {
    'BLEU': 1,
    'BLANC': 2,
    'ROUGE': 3
}

# Créer la colonne 'tempo_num'
df['Type de jour TEMPO'].map(tempo_mapping)

df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M').dt.time

# Fonction pour attribuer une période
def get_periode_jour(time_obj):
    if dt.time(0, 0) <= time_obj < dt.time(5, 0):
        return 'nuit'
    elif dt.time(5, 0) <= time_obj < dt.time(12, 0):
        return 'matin'
    elif dt.time(12, 0) <= time_obj < dt.time(14, 0):
        return 'midi'
    elif dt.time(14, 0) <= time_obj < dt.time(18, 0):
        return 'après-midi'
    else:
        return 'soir'

# Appliquer la fonction pour créer la colonne
df['periode_jour'] = df['Heures'].apply(get_periode_jour)

# Fonction de classification en période (4 valeurs)
def encoder_periode_jour(time_obj):
    if dt.time(0, 0) <= time_obj < dt.time(6, 0):
        return 1  # nuit
    elif dt.time(6, 0) <= time_obj < dt.time(12, 0):
        return 2  # matin
    elif dt.time(12, 0) <= time_obj < dt.time(18, 0):
        return 3  # après-midi (inclut midi)
    else:
        return 4  # soir

# Créer la colonne encodée
df['periode_jour_code'] = df['Heures'].apply(encoder_periode_jour)

# Assurez-vous que cette fonction est définie avant de l'appeler
def preparer_donnees_tempo(df):
    # Ajoutez ici le code de la fonction si nécessaire
    return df

df = preparer_donnees_tempo(df)
df.to_csv("df_processed.csv", sep=",", index=False, encoding="utf-8")

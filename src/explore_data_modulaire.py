
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def charger_donnees(path='data/processed/df_filtre.csv'):
    """Charge le fichier de données brutes filtrées."""
    try:
        df = pd.read_csv(path, sep='\t', encoding='latin1')
        print(f"Chargement réussi : {path}")
        return df
    except FileNotFoundError:
        print(f"Erreur : le fichier {path} est introuvable.")
        return None

def afficher_apercu(df):
    """Affiche un aperçu de base des données."""
    print(df.head())
    print(df.info())
    print(df.describe())

def plot_boxplot(df, col_x, col_y, title, output_path):
    """Génère et enregistre un boxplot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=col_x, y=col_y)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")

def generer_boxplots(df):
    """Génère tous les boxplots nécessaires pour l'exploration."""
    os.makedirs('../outputs/figures', exist_ok=True)
    plot_boxplot(df, 'Weekend', 'Consommation', 'Consommation selon le week-end', '../outputs/figures/boxplot_weekend.png')
    plot_boxplot(df, 'Saison', 'Consommation', 'Consommation selon la saison', '../outputs/figures/boxplot_saison.png')
    plot_boxplot(df, 'Jour', 'Consommation', 'Consommation selon les jours de la semaine', '../outputs/figures/boxplot_jour_semaine.png')

def pipeline_exploration():
    """Pipeline complet pour l’exploration des données."""
    df = charger_donnees()
    if df is not None:
        afficher_apercu(df)
        generer_boxplots(df)

if __name__ == "__main__":
    pipeline_exploration()


import pandas as pd
import matplotlib.pyplot as plt
import os

def charger_performances(fichier='outputs/performances.csv'):
    """Charge le fichier CSV contenant les performances des modèles."""
    if os.path.exists(fichier):
        return pd.read_csv(fichier)
    else:
        print(f"Fichier {fichier} introuvable.")
        return None

def comparer_performances(df, output_path="outputs/figures/comparaison_modeles.png"):
    """Génère un graphique de comparaison des performances des modèles."""
    if df is None or df.empty:
        print("Aucune donnée à visualiser.")
        return

    metrics = ['MAE', 'RMSE', 'R2']
    models = df['Modèle']
    x = range(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar([p + i * width for p in x], df[metric], width, label=metric)

    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(models)
    ax.set_ylabel("Scores")
    ax.set_title("Comparaison des performances des modèles")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé dans {output_path}")

def pipeline_comparaison():
    df = charger_performances()
    comparer_performances(df)

if __name__ == "__main__":
    pipeline_comparaison()

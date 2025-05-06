
import pandas as pd
import os

def enregistrer_performances_csv(nom_modele, mae, rmse, r2, fichier='outputs/performances.csv'):
    """
    Enregistre les performances d'un modèle dans un fichier CSV.
    Si le fichier existe, il ajoute une ligne. Sinon, il le crée.
    """
    os.makedirs(os.path.dirname(fichier), exist_ok=True)
    
    nouvelle_ligne = {
        'Modèle': nom_modele,
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 4)
    }

    try:
        df = pd.read_csv(fichier)
        df = pd.concat([df, pd.DataFrame([nouvelle_ligne])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([nouvelle_ligne])

    df.to_csv(fichier, index=False)
    print(f"Performance du modèle '{nom_modele}' enregistrée dans {fichier}")

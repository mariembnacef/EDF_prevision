import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def process_consumption_data(path_annuel, path_calendar, output_path=None):
    """
    Process and clean electricity consumption data by integrating annual data with calendar information.
    
    Args:
        path_annuel (str): Path to the folder containing annual consumption data files
        path_calendar (str): Path to the folder containing calendar data files
        output_path (str, optional): Path where the processed data will be saved. If None, no file is saved.
        
    Returns:
        pd.DataFrame: The processed and cleaned dataframe
    """
    
    # Helper function to read dataframes
    def read_df(path):
        df = pd.read_csv(path, sep="\t", encoding="latin1", index_col=False)
        return df

    # Helper function to merge files from a folder
    def merge_from_folder(folder_path):
        """
        Uses read_df to read and merge all CSV or TXT files from a folder.
        """
        all_files = [f for f in os.listdir(folder_path)]
        dfs = []
        
        for file in all_files:
            full_path = os.path.join(folder_path, file)
            df = read_df(full_path)
            dfs.append(df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    
    # Load and merge data
    annuel = merge_from_folder(path_annuel)
    calendrier = merge_from_folder(path_calendar)
    
    # Check for duplicates
    doublons = annuel[annuel.duplicated()]
    print(f"Found {len(doublons)} duplicate rows in annual data")
    
    # Process calendar data
    calendrier["Date"] = pd.to_datetime(calendrier["Date"], errors='coerce')
    calendrier = calendrier.dropna(subset=["Date"]).reset_index(drop=True)
    
    # Process annual data
    annuel.replace("ND", pd.NA, inplace=True)
    annuel = annuel[annuel["Périmètre"] == "France"]
    annuel.replace("ND", np.nan, inplace=True)
    
    # Merge datasets by date
    def fusionner_par_date(annuel, calendrier, type_jointure="inner"):
        """
        Merges two DataFrames on the 'Date' column after converting to datetime.
        """
        # Convert to datetime
        annuel['Date'] = pd.to_datetime(annuel['Date'], errors='coerce')
        calendrier['Date'] = pd.to_datetime(calendrier['Date'], errors='coerce')

        # Remove rows with invalid dates
        annuel = annuel.dropna(subset=['Date'])
        calendrier = calendrier.dropna(subset=['Date'])

        # Merge
        fusion = pd.merge(annuel, calendrier, on='Date', how=type_jointure)
        return fusion
    
    df_final = fusionner_par_date(annuel, calendrier, type_jointure="inner")
    
    # Keep only useful columns
    def garder_colonnes_utiles(df):
        """
        Keep only essential columns in the DataFrame.
        """
        colonnes_a_garder = [
            'Type de jour TEMPO',
            'Date',
            'Heures',
            'Prévision J',
            'Prévision J-1',
            'Consommation'
        ]
        
        # Keep only columns that exist in the DataFrame
        colonnes_presentes = [col for col in colonnes_a_garder if col in df.columns]
        
        return df[colonnes_presentes].copy()
    
    df_v1 = garder_colonnes_utiles(df_final)
    
    # Remove NaN in odd quarter-hours
    def supprimer_nan_quart_impair(df):
        """
        Removes rows where 'Heures' is at :15 or :45 and 'Consommation' is NaN.
        """
        # Extract minutes as integers (e.g., 15, 30, 45, ...)
        df['Minutes'] = df['Heures'].astype(str).str.slice(3, 5).astype(int)

        # Identify rows to remove: minutes = 15 or 45 AND consumption = NaN
        condition_suppr = df['Minutes'].isin([15, 45]) & df['Consommation'].isna()

        # Remove these rows
        df_filtré = df[~condition_suppr].copy()

        # Cleanup: remove temporary column
        df_filtré.drop(columns='Minutes', inplace=True)

        return df_filtré
    
    df_filtre = supprimer_nan_quart_impair(df_v1)
    
    # Add temporal information
    def ajouter_infos_temporelles(df):
        """
        Adds 'Jour', 'Weekend', and 'Saison' columns based on the 'Date' column.
        """
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Day of the week - Without using locale
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

        # Weekend (Saturday = 5, Sunday = 6)
        df['Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)

        # Month to determine seasons
        mois = df['Date'].dt.month

        # Function to determine season
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
    
    # Add temporal information
    df_filtre = ajouter_infos_temporelles(df_filtre)
    
    # Compare stats before and after filtering (just for info)
    def get_comparison_stats(df_original, df_nettoye):
        total_original = len(df_original)
        total_nettoye = len(df_nettoye)
        nan_original = df_original['Consommation'].isna().sum()
        nan_nettoye = df_nettoye['Consommation'].isna().sum()
        
        stats = {
            "rows_before": total_original,
            "rows_after": total_nettoye,
            "nan_before": nan_original,
            "nan_after": nan_nettoye,
            "rows_removed": total_original - total_nettoye
        }
        
        print(f"Stats: {stats}")
        return stats
    
    # Get comparison statistics
    stats = get_comparison_stats(df_v1, df_filtre)
    
    # Save to file if output path is provided
    if output_path:
        df_filtre.to_csv(output_path, sep="\t", index=False, encoding="latin1")
        print(f"Processed data saved to {output_path}")
    
    return df_filtre


# Example usage:
if __name__ == "__main__":
    # Define paths
    path_annuel = "../data/raw/annuel"
    path_calendar = "../data/raw/calendar"
    output_path = "../data/df_filtre.csv"
    
    # Process data
    processed_df = process_consumption_data(path_annuel, path_calendar, output_path)
    print(f"Processed dataframe shape: {processed_df.shape}")
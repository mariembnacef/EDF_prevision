#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline complète de traitement des données de consommation électrique et modélisation
Ce script orchestre le processus complet en trois étapes :
1. Téléchargement des fichiers de données à partir de RTE
2. Préparation des données
3. Entraînement et évaluation du modèle
"""

import os
import time
import sys
import subprocess
from datetime import datetime

# Impormodues personnalisés
import processing_data
import train_test

def main():
    """
    Fonction principale qui exécute la pipeline complète
    en utilisant les chemins par défaut du projet
    """
    start_time = time.time()
    
    # Définition des chemins directement basés sur la structure du projet
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    path_annuel = os.path.join(raw_dir, "annuel")
    path_calendar = os.path.join(raw_dir, "calendar")
    processed_data_path = os.path.join(data_dir, "processed", "df_filtre.csv")
    model_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(model_dir, f"xgboost_conso_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    # Création des répertoires nécessaires
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Paramètres de configuration
    do_download = True    # Mettre à False pour sauter l'étape de téléchargement
    do_processing = True  # Mettre à False pour sauter l'étape de prétraitement
    verbose = True        # Afficher plus de détails pendant l'entraînement
    
    # ÉTAPE 0: Téléchargement des fichiers (optionnel)
    if do_download:
        print("🔄 ÉTAPE 0: Téléchargement des fichiers depuis RTE...")
        try:
            # Exécution du script de téléchargement
            download_script = os.path.join(base_dir, "download_file.py")
            result = subprocess.run([sys.executable, download_script], check=True)
            
            if result.returncode == 0:
                print("✅ Téléchargement terminé avec succès")
                
                # Déplacer les fichiers extraits vers les bons répertoires
                download_extracted = os.path.join(base_dir, "rte_extracted")
                
                # Créer les sous-répertoires s'ils n'existent pas
                os.makedirs(path_annuel, exist_ok=True)
                os.makedirs(path_calendar, exist_ok=True)
                
                # Déplacer les fichiers annuels
                src_annual = os.path.join(download_extracted, "annuel")
                if os.path.exists(src_annual):
                    os.system(f'cp -r {src_annual}/* {path_annuel}/')
                    print(f"✅ Données annuelles copiées vers {path_annuel}")
                
                # Déplacer les fichiers calendrier (inclut les données TEMPO)
                src_calendar = os.path.join(download_extracted, "calendrier")
                src_tempo = os.path.join(download_extracted, "tempo")
                
                if os.path.exists(src_calendar):
                    os.system(f'cp -r {src_calendar}/* {path_calendar}/')
                    print(f"✅ Données de calendrier copiées vers {path_calendar}")
                
                if os.path.exists(src_tempo):
                    os.system(f'cp -r {src_tempo}/* {path_calendar}/')
                    print(f"✅ Données TEMPO copiées vers {path_calendar}")
            else:
                print("⚠️ Le téléchargement a retourné un code d'erreur")
        except Exception as e:
            print(f"❌ ERREUR lors du téléchargement des données: {str(e)}")
            print("⚠️ Tentative de continuer avec les données existantes...")
    else:
        print("🔄 ÉTAPE 0: Téléchargement ignoré (utilisation des données existantes)")
    
    # ÉTAPE 1: Traitement des données (optionnel)
    if do_processing:
        print("\n🔄 ÉTAPE 1: Préparation et nettoyage des données...")
        try:
            df = processing_data.process_consumption_data(
                path_annuel, 
                path_calendar, 
                processed_data_path
            )
            print(f"✅ Données traitées et sauvegardées dans {processed_data_path}")
            print(f"   → {df.shape[0]} lignes × {df.shape[1]} colonnes")
        except Exception as e:
            print(f"❌ ERREUR lors du traitement des données: {str(e)}")
            return 1
    else:
        print("\n🔄 ÉTAPE 1: Prétraitement ignoré (utilisation des données existantes)")
        
        if not os.path.exists(processed_data_path):
            print(f"❌ ERREUR: Le fichier de données {processed_data_path} n'existe pas")
            return 1
    
    # ÉTAPE 2: Préparation des données pour la modélisation
    print("\n🔄 ÉTAPE 2: Préparation des données pour la modélisation...")
    try:
        X, y, df_model = train_test.preparer_donnees(processed_data_path)
        print(f"✅ Données préparées: {X.shape[0]} échantillons avec {X.shape[1]} caractéristiques")
    except Exception as e:
        print(f"❌ ERREUR lors de la préparation des données pour la modélisation: {str(e)}")
        return 1
    
    # ÉTAPE 3: Entraînement du modèle
    print("\n🔄 ÉTAPE 3: Entraînement du modèle et optimisation des hyperparamètres...")
    try:
        best_model, X_train, X_val, y_train, y_val = train_test.entrainer_modele(X, y, verbose=verbose)
        print("✅ Modèle entraîné avec succès")
    except Exception as e:
        print(f"❌ ERREUR lors de l'entraînement du modèle: {str(e)}")
        return 1
    
    # ÉTAPE 4: Évaluation du modèle
    print("\n🔄 ÉTAPE 4: Évaluation des performances du modèle...")
    try:
        resultats = train_test.evaluer_modele(best_model, X_train, X_val, y_train, y_val)
        status = "⚠️ ATTENTION: Overfitting détecté" if resultats['overfitting'] else "✅ Pas d'overfitting significatif"
        print(status)
    except Exception as e:
        print(f"❌ ERREUR lors de l'évaluation du modèle: {str(e)}")
        return 1
    
    # ÉTAPE 5: Sauvegarde du modèle
    print("\n🔄 ÉTAPE 5: Sauvegarde du modèle...")
    try:
        chemin_modele = train_test.sauvegarder_modele(best_model, model_path)
        print(f"✅ Modèle sauvegardé sous '{chemin_modele}'")
    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde du modèle: {str(e)}")
        return 1
    
    # Affichage du temps d'exécution
    execution_time = time.time() - start_time
    print(f"\n✨ Pipeline terminée en {execution_time:.2f} secondes ({execution_time/60:.2f} minutes)")
    print(f"📊 Modèle final - R²: {resultats['r2_val']:.4f}, RMSE: {resultats['rmse_val']:.2f}")
    print(f"💾 Modèle sauvegardé: {model_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline simplifiée de traitement des données de consommation électrique et modélisation
Ce script orchestre le processus complet en deux étapes :
1. Préparation des données (optionnel, utilise le chemin par défaut des données brutes)
2. Entraînement et évaluation du modèle
"""

import os
import time
from datetime import datetime

# Import des modules personnalisés
import processing_data
import train_test

def main():
    """
    Fonction principale simplifiée qui exécute la pipeline complète
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
    
    # Paramètres de configuration
    do_processing = True  # Mettre à False pour sauter l'étape de prétraitement
    verbose = True        # Afficher plus de détails pendant l'entraînement
    
    # ÉTAPE 1: Traitement des données (optionnel)
    if do_processing:
        print("🔄 ÉTAPE 1: Préparation et nettoyage des données...")
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
        print("🔄 ÉTAPE 1: Prétraitement ignoré (utilisation des données existantes)")
        
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
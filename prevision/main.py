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
import sys
import time
import subprocess
from datetime import datetime

import mlflow
import train_test


def main():
    """
    Fonction principale qui exécute la pipeline complète
    en utilisant MLflow pour le suivi des expériences.
    """
    start_time = time.time()

    # ── 0) Configuration MLflow ─────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("conso-electrique-xgboost")

    # Définition des chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    extracted_dir = os.path.join(base_dir, "rte_extracted")
    annuel_dir = os.path.join(extracted_dir, "annuel")
    calendar_dir = os.path.join(extracted_dir, "calendar")
    processed_data_path = os.path.join(data_dir, "processed", "df_filtre.csv")
    model_dir = os.path.join(base_dir, "models")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f"xgboost_conso_model_{timestamp}.pkl")

    # Création des répertoires
    for path in [os.path.dirname(processed_data_path), model_dir, raw_dir, extrait_dir := extracted_dir, annuel_dir, calendar_dir]:
        os.makedirs(path, exist_ok=True)

    # Options
    do_download = True
    do_processing = True
    verbose = True
    mode_rapide = True  # Activation du mode rapide

    # Démarrage de la run MLflow
    with mlflow.start_run(run_name=f"xgboost_{timestamp}"):
        # Log des paramètres initiaux
        mlflow.log_params({
            "timestamp": timestamp,
            "do_download": do_download,
            "do_processing": do_processing,
            "verbose": verbose,
            "mode_rapide": mode_rapide
        })

        # ── ÉTAPE 0: Téléchargement ────────────────────────────────────────
        if do_download:
            print("🔄 ÉTAPE 0: Téléchargement des fichiers depuis RTE...")
            # Import du module de téléchargement ou exécution du script
            try:
                import download_file
                download_file.main()
                download_status = "success"
            except Exception:
                # Fallback à l'appel du script externe
                script_path = os.path.join(base_dir, "download_file.py")
                try:
                    subprocess.run([sys.executable, script_path], check=True)
                    download_status = "success"
                except subprocess.CalledProcessError as e:
                    download_status = "failed"
                    error_msg = str(e)

            mlflow.log_param("download_status", download_status)
            if download_status == "success":
                print("✅ Téléchargement terminé avec succès")
            else:
                mlflow.log_param("download_error", error_msg)
                print(f"❌ ERREUR lors du téléchargement: {error_msg}")
                print("⚠️ Continuation avec les données déjà présentes...")
        else:
            mlflow.log_param("download_status", "skipped")
            print("🔄 ÉTAPE 0: Ignoré (données existantes)")

        # ── ÉTAPE 1: Prétraitement des données ─────────────────────────────
        if do_processing:
            print("\n🔄 ÉTAPE 1: Prétraitement des données...")
            try:
                # Appel d'une fonction de préparation si disponible
                import prepare_data
                prepare_data.run(raw_dir, processed_data_path, verbose=verbose)
                processing_status = "success"
            except ImportError:
                # Cas où la préparation est intégrée dans train_test
                processing_status = "success"
            except Exception as e:
                processing_status = "failed"
                proc_error = str(e)

            mlflow.log_param("processing_status", processing_status)
            if processing_status == "success":
                print("✅ Prétraitement des données terminé avec succès")
            else:
                mlflow.log_param("processing_error", proc_error)
                print(f"❌ ERREUR lors du prétraitement: {proc_error}")
                print("⚠️ Continuation avec les données existantes...")
        else:
            mlflow.log_param("processing_status", "skipped")
            print("🔄 ÉTAPE 1: Ignoré (prétraitement existant)")

        # ── ÉTAPE 2 & 3: Entraînement et Évaluation ────────────────────────
        try:
            print("\n🔄 ÉTAPE 2 & 3: Entraînement et évaluation du modèle...")
            best_model, resultats = train_test.executer_pipeline_complete(
                processed_data_path,
                model_path,
                verbose=verbose,
                mode_rapide=mode_rapide
            )

            # Log des métriques
            mlflow.log_metrics({
                "final_rmse_val": resultats.get('rmse_val'),
                "final_r2_val": resultats.get('r2_val')
            })
            mlflow.log_param("overfitting_detected", resultats.get('overfitting', False))

            # Durée totale
            total_time = time.time() - start_time
            mlflow.log_metric("pipeline_total_time_seconds", total_time)

            print(f"\n✨ Pipeline terminée en {total_time:.2f}s ({total_time/60:.2f}min)")
            print(f"📊 R² validation: {resultats['r2_val']:.4f}, RMSE validation: {resultats['rmse_val']:.2f}")
            print(f"💾 Modèle final sauvegardé sous: {model_path}")

            return 0
        except Exception as e:
            error_msg = str(e)
            mlflow.log_param("error", error_msg)
            print(f"❌ ERREUR dans la pipeline: {error_msg}")
            return 1


if __name__ == "__main__":
    sys.exit(main())

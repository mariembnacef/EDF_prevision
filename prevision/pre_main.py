#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de traitement des données de consommation électrique et modélisation avec Prefect
Ce script orchestre le processus complet en trois étapes avec Prefect:
1. Téléchargement des fichiers de données à partir de RTE
2. Préparation des données
3. Entraînement et évaluation du modèle
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple, Any

import mlflow
import prefect
from prefect import task, flow
from prefect.context import get_run_context

# Import des modules nécessaires
# Ces importations seront tentées mais le code continuera même si certains modules manquent
train_test = None
download_file = None
processing_data = None  # Changé de prepare_data à processing_data

try:
    import train_test
except ImportError:
    print("Module train_test non trouvé. Certaines fonctionnalités seront limitées.")

try:
    import download_file
except ImportError:
    print("Module download_file non trouvé. Le téléchargement sera désactivé.")

try:
    import processing_data  # Changé de preparer_donnees à processing_data
except ImportError:
    print("Module processing_data non trouvé. Le prétraitement intégré sera utilisé si disponible.")


@task(name="Configuration des chemins", description="Définit les chemins utilisés dans le pipeline")
def setup_directories() -> Dict[str, str]:
    """
    Configuration des répertoires pour le pipeline.
    
    Returns:
        Dict[str, str]: Dictionnaire contenant tous les chemins nécessaires
    """
    # Définition des chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    extracted_dir = os.path.join(base_dir, "rte_extracted")
    annuel_dir = os.path.join(extracted_dir, "annuel")
    calendar_dir = os.path.join(extracted_dir, "calendar")
    processed_dir = os.path.join(data_dir, "processed")
    processed_data_path = os.path.join(processed_dir, "df_filtre.csv")
    model_dir = os.path.join(base_dir, "models")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f"xgboost_conso_model_{timestamp}.pkl")
    
    # Création des répertoires
    for path in [raw_dir, extracted_dir, annuel_dir, calendar_dir, processed_dir, model_dir]:
        os.makedirs(path, exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "extracted_dir": extracted_dir,
        "annuel_dir": annuel_dir,
        "calendar_dir": calendar_dir,
        "processed_data_path": processed_data_path,
        "model_dir": model_dir,
        "model_path": model_path,
        "timestamp": timestamp
    }


@task(name="Configuration MLflow", description="Configure MLflow pour le tracking des expériences")
def setup_mlflow() -> str:
    """
    Configure MLflow pour le suivi des expériences.
    
    Returns:
        str: URI de tracking MLflow
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("conso-electrique-xgboost")
    return tracking_uri


@task(name="Téléchargement des données", 
      description="Téléchargement des fichiers depuis RTE",
      retries=2,
      retry_delay_seconds=30)
def download_data(paths: Dict[str, str]) -> Dict[str, str]:
    """
    Téléchargement des fichiers de données depuis RTE.
    
    Args:
        paths (Dict[str, str]): Dictionnaire des chemins
        
    Returns:
        Dict[str, str]: Statut du téléchargement et métadonnées
    """
    logger = prefect.get_run_logger()
    logger.info("🔄 Téléchargement des fichiers depuis RTE...")
    
    try:
        download_file.main()
        logger.info("✅ Téléchargement terminé avec succès")
        return {"status": "success", "message": "Téléchargement réussi"}
    except Exception as e:
        if download_file is None:
            logger.error("Module download_file non disponible")
            return {"status": "failed", "error": "Module manquant", "message": "Module download_file non disponible"}
        else:
            logger.error(f"❌ ERREUR lors du téléchargement: {str(e)}")
            logger.warning("⚠️ Continuation avec les données déjà présentes...")
            return {"status": "failed", "error": str(e), "message": "Échec du téléchargement"}


@task(name="Prétraitement des données", 
      description="Préparation et nettoyage des données",
      retries=1)
def process_data(paths: Dict[str, str], verbose: bool = True) -> Dict[str, str]:
    """
    Prétraitement des données téléchargées.
    
    Args:
        paths (Dict[str, str]): Dictionnaire des chemins
        verbose (bool): Activation des logs détaillés
        
    Returns:
        Dict[str, str]: Statut du prétraitement et métadonnées
    """
    logger = prefect.get_run_logger()
    logger.info("🔄 Prétraitement des données...")
    
    try:
        # Modifié pour utiliser processing_data.process_consumption_data
        if processing_data is not None:
            # Appel de process_consumption_data avec les bons paramètres
            processing_data.process_consumption_data(
                paths["annuel_dir"], 
                paths["calendar_dir"], 
                paths["processed_data_path"]
            )
            logger.info("✅ Prétraitement des données terminé avec succès")
            return {"status": "success", "message": "Prétraitement réussi", "output_path": paths["processed_data_path"]}
        else:
            # Essaie d'utiliser le prétraitement intégré dans train_test si processing_data n'est pas disponible
            logger.info("Module processing_data non disponible, vérification si le prétraitement est intégré dans train_test...")
            if hasattr(train_test, 'prepare_data') or hasattr(train_test, 'preprocess_data'):
                preprocess_fn = getattr(train_test, 'prepare_data', None) or getattr(train_test, 'preprocess_data', None)
                preprocess_fn(paths["raw_dir"], paths["processed_data_path"], verbose=verbose)
                logger.info("✅ Prétraitement intégré terminé avec succès")
                return {"status": "success", "message": "Prétraitement intégré réussi", "output_path": paths["processed_data_path"]}
            else:
                logger.warning("Aucune fonction de prétraitement disponible. Vérifiez les données prétraitées existantes.")
                return {"status": "skipped", "message": "Aucun prétraitement disponible"}
    except Exception as e:
        logger.error(f"❌ ERREUR lors du prétraitement: {str(e)}")
        logger.warning("⚠️ Continuation avec les données prétraitées existantes...")
        return {"status": "failed", "error": str(e), "message": "Échec du prétraitement"}


@task(name="Entraînement et évaluation", 
      description="Entraînement du modèle XGBoost et évaluation des performances")
def train_and_evaluate(
    paths: Dict[str, str], 
    verbose: bool = True, 
    mode_rapide: bool = True
) -> Dict[str, Any]:
    """
    Entraînement et évaluation du modèle.
    
    Args:
        paths (Dict[str, str]): Dictionnaire des chemins
        verbose (bool): Activation des logs détaillés
        mode_rapide (bool): Utilisation du mode d'entraînement rapide
        
    Returns:
        Dict[str, Any]: Résultats de l'entraînement et métriques
    """
    logger = prefect.get_run_logger()
    logger.info("🔄 Entraînement et évaluation du modèle...")
    
    try:
        if train_test is None:
            logger.error("Module train_test non disponible, impossible de poursuivre l'entraînement")
            return {"status": "failed", "error": "Module manquant", "message": "Module train_test requis non disponible"}
            
        best_model, resultats = train_test.executer_pipeline_complete(
            paths["processed_data_path"],
            paths["model_path"],
            verbose=verbose,
            mode_rapide=mode_rapide
        )
        
        logger.info(f"✅ Entraînement terminé avec succès")
        logger.info(f"📊 R² validation: {resultats['r2_val']:.4f}, RMSE validation: {resultats['rmse_val']:.2f}")
        
        # Ajout des métriques pour Prefect
        return {
            "status": "success",
            "model_path": paths["model_path"],
            "rmse_val": resultats.get('rmse_val'),
            "r2_val": resultats.get('r2_val'),
            "overfitting": resultats.get('overfitting', False)
        }
    except Exception as e:
        logger.error(f"❌ ERREUR lors de l'entraînement: {str(e)}")
        return {"status": "failed", "error": str(e)}


@flow(name="Pipeline Consommation Électrique",
      description="Pipeline complète d'analyse de la consommation électrique")
def pipeline_consommation_electrique(
    do_download: bool = True,
    do_processing: bool = True,
    verbose: bool = True,
    mode_rapide: bool = True
) -> Dict[str, Any]:
    """
    Flow principal Prefect qui orchestre le pipeline complet.
    
    Args:
        do_download (bool): Activer le téléchargement des données
        do_processing (bool): Activer le prétraitement des données
        verbose (bool): Activer les logs détaillés
        mode_rapide (bool): Utiliser le mode d'entraînement rapide
        
    Returns:
        Dict[str, Any]: Résultats globaux du pipeline
    """
    logger = prefect.get_run_logger()
    start_time = time.time()
    
    # Récupération du contexte du flow
    try:
        context = get_run_context()
        run_id = context.flow_run.id if hasattr(context, 'flow_run') else "unknown"
    except Exception:
        run_id = "local-run"
    
    # 0. Configuration des chemins et MLflow
    paths = setup_directories()
    tracking_uri = setup_mlflow()
    
    # Démarrage de la run MLflow
    with mlflow.start_run(run_name=f"xgboost_{paths['timestamp']}"):
        # Log des paramètres initiaux
        mlflow.log_params({
            "timestamp": paths["timestamp"],
            "do_download": do_download,
            "do_processing": do_processing,
            "verbose": verbose,
            "mode_rapide": mode_rapide,
            "prefect_run_id": run_id
        })
        
        # 1. Téléchargement
        download_result = {"status": "skipped"}
        if do_download:
            download_result = download_data(paths)
            mlflow.log_param("download_status", download_result["status"])
            if download_result["status"] == "failed":
                mlflow.log_param("download_error", download_result.get("error", "unknown"))
        else:
            logger.info("🔄 Téléchargement ignoré (données existantes)")
            mlflow.log_param("download_status", "skipped")
        
        # 2. Prétraitement
        processing_result = {"status": "skipped"}
        if do_processing:
            processing_result = process_data(paths, verbose)
            mlflow.log_param("processing_status", processing_result["status"])
            if processing_result["status"] == "failed":
                mlflow.log_param("processing_error", processing_result.get("error", "unknown"))
        else:
            logger.info("🔄 Prétraitement ignoré (données prétraitées existantes)")
            mlflow.log_param("processing_status", "skipped")
        
        # 3. Entraînement et évaluation
        training_result = train_and_evaluate(paths, verbose, mode_rapide)
        
        if training_result["status"] == "success":
            # Log des métriques dans MLflow
            mlflow.log_metrics({
                "final_rmse_val": training_result["rmse_val"],
                "final_r2_val": training_result["r2_val"]
            })
            mlflow.log_param("overfitting_detected", training_result.get("overfitting", False))
            
            # Enregistrement du chemin du modèle
            mlflow.log_param("model_path", training_result["model_path"])
        else:
            mlflow.log_param("training_error", training_result.get("error", "unknown"))
        
        # Durée totale
        total_time = time.time() - start_time
        mlflow.log_metric("pipeline_total_time_seconds", total_time)
        
        logger.info(f"\n✨ Pipeline terminée en {total_time:.2f}s ({total_time/60:.2f}min)")
        
        if training_result["status"] == "success":
            logger.info(f"📊 R² validation: {training_result['r2_val']:.4f}, RMSE validation: {training_result['rmse_val']:.2f}")
            logger.info(f"💾 Modèle final sauvegardé sous: {paths['model_path']}")
        
        # Résultat global du pipeline
        return {
            "success": training_result["status"] == "success",
            "download": download_result,
            "processing": processing_result,
            "training": training_result,
            "total_time": total_time,
            "paths": paths
        }


if __name__ == "__main__":
    # Exécution du flow Prefect
    pipeline_consommation_electrique()
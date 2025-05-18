import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Query
from dateutil import parser as dtparser
from datetime import date, time, datetime
from dateutil.easter import easter
import pandas as pd
import uvicorn

def get_best_model_path_from_mlflow(
    experiment_name: str,
    metric_name: str = "r2_val",
    maximize: bool = True,
    artifact_subpath: str = "model"
) -> str:
    """
    Recherche dans MLflow l'expérimentation `experiment_name`,
    récupère tous les runs, filtre ceux qui ont la métrique `metric_name`,
    les trie (DESC si maximize else ASC) et télécharge le meilleur artefact.
    """
    # 1. Connexion à MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 2. Récupération de l'expérience
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Expérience MLflow '{experiment_name}' introuvable")
    exp_id = exp.experiment_id

    # 3. Lister tous les runs (pas de filter SQL)
    runs = client.search_runs(experiment_ids=[exp_id], max_results=1000)

    # 4. Filtrer ceux qui ont la métrique
    valid_runs = [
        r for r in runs
        if metric_name in r.data.metrics and r.data.metrics[metric_name] is not None
    ]
    if not valid_runs:
        raise RuntimeError(f"Aucun run avec la métrique '{metric_name}' trouvé")

    # 5. Trier et prendre le meilleur
    valid_runs.sort(
        key=lambda r: r.data.metrics[metric_name],
        reverse=bool(maximize)
    )
    best_run = valid_runs[0]

    # 6. Télécharger l'artefact
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id,
        artifact_path=artifact_subpath
    )
    # 7. Trouver un .pkl ou .joblib
    for fname in os.listdir(local_dir):
        if fname.endswith((".pkl", ".joblib")):
            return os.path.join(local_dir, fname)

    raise FileNotFoundError(f"Aucun .pkl/.joblib dans '{local_dir}'")

# --- Chargement du modèle au démarrage ---
EXPERIMENT_NAME = "conso-electrique-xgboost"
try:
    MODEL_PATH = get_best_model_path_from_mlflow(
        EXPERIMENT_NAME,
        metric_name="r2_val",
        maximize=True,
        artifact_subpath="model"
    )
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis MLflow (run optimal) : {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle depuis MLflow : {e}")

app = FastAPI(title="API Prédiction Consommation (ISO datetime)")

# --- Helpers pour features (inchangés) ---
def infer_season(m: int) -> int:
    if m in (9,10,11): return 1
    if m in (12,1,2):  return 2
    if m in (3,4,5):   return 3
    return 4

def is_holiday(d: date) -> int:
    fixed = {
        date(d.year,1,1), date(d.year,5,1), date(d.year,5,8),
        date(d.year,7,14), date(d.year,8,15),
        date(d.year,11,1), date(d.year,11,11), date(d.year,12,25)
    }
    e = easter(d.year)
    movable = {
        e + pd.Timedelta(days=1),
        e + pd.Timedelta(days=39),
        e + pd.Timedelta(days=50)
    }
    return int(d in fixed or d in movable)

@app.get("/predict/")
def predict(
    datetime_iso: str = Query(
        ...,
        description="Datetime ISO8601, ex. 2025-05-14T22:59:07.941Z"
    )
):
    # Parse
    try:
        dt_obj = dtparser.isoparse(datetime_iso)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de parser le datetime : {e}")

    d = dt_obj.date()
    h = dt_obj.time()

    # Création des features
    feat = {
        "Weekend": int(d.weekday() >= 5),
        "mois": d.month,
        "annee": d.year,
        "jour_semaine": d.weekday() + 1,
        "saison_num": infer_season(d.month),
        "tempo_num": 0,
        "periode_jour_code": (
            1 if h < time(5) else
            2 if h < time(12) else
            3 if h < time(18) else
            4
        ),
        "jour_ferie": is_holiday(d),
        "Heures_float": h.hour + h.minute / 60 + h.second / 3600,
    }
    for lag in (1,2,3,4):
        feat[f"lag_{lag}"] = 0.0

    X = pd.DataFrame([feat])

    # Prédiction
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return {
        "datetime": datetime_iso,
        "prediction": float(pred)
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)

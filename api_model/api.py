import os
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from dateutil import parser as dtparser
from datetime import date, time
import pandas as pd
import numpy as np
import joblib
from dateutil.easter import easter
import uvicorn

def get_latest_model_path(models_dir: str) -> str:
    """
    Parcourt models_dir, extrait tous les timestamps de la forme YYYYMMDD_HHMMSS
    dans chaque nom de fichier, et renvoie le chemin du fichier dont le dernier
    timestamp (le plus grand) est le plus récent.
    """
    candidates = []
    ts_pattern = re.compile(r"(\d{8}_\d{6})")
    for fname in os.listdir(models_dir):
        # on trouve toutes les chaînes au format YYYYMMDD_HHMMSS
        matches = ts_pattern.findall(fname)
        if matches:
            # on convertit chaque match en datetime, on prend le max
            latest_ts = max(datetime.strptime(ts, "%Y%m%d_%H%M%S") for ts in matches)
            candidates.append((latest_ts, fname))
    if not candidates:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {models_dir}")
    # trier par timestamp et prendre le plus récent
    _, latest_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(models_dir, latest_fname)

# --- Chargement du modèle au démarrage ---
MODELS_DIR = "models"
try:
    MODEL_PATH = get_latest_model_path(MODELS_DIR)
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle : {e}")

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
        description="Datetime au format ISO8601, ex. 2014-09-01T00:00:00Z ou 2025-05-14T22:59:07.941Z"
    )
):
    try:
        dt_obj = dtparser.isoparse(datetime_iso)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de parser le datetime : {e}")

    d = dt_obj.date()
    h = dt_obj.time()

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

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return {
        "datetime": datetime_iso,
        "prediction": float(pred)
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)

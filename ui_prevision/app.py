import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
from datetime import datetime, date, time as dt_time
from dateutil.easter import easter
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient
import requests

# --- Fonction utilitaire MLflow ---
def get_best_model_path_from_mlflow(
    experiment_name: str,
    metric_name: str = "r2_val",
    maximize: bool = True,
    artifact_subpath: str = "model"
) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Expérience MLflow '{experiment_name}' introuvable")
    exp_id = exp.experiment_id

    runs = client.search_runs(experiment_ids=[exp_id], max_results=1000)
    valid_runs = [r for r in runs
                  if metric_name in r.data.metrics and r.data.metrics[metric_name] is not None]
    if not valid_runs:
        raise RuntimeError(f"Aucun run avec la métrique '{metric_name}' trouvé")

    valid_runs.sort(key=lambda r: r.data.metrics[metric_name], reverse=bool(maximize))
    best_run = valid_runs[0]
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id,
        artifact_path=artifact_subpath
    )
    for fname in os.listdir(local_dir):
        if fname.endswith((".pkl", ".joblib")):
            return os.path.join(local_dir, fname)
    raise FileNotFoundError(f"Aucun .pkl/.joblib dans '{local_dir}'")

# --- Chargement du (meilleur) modèle MLflow ou fallback local ---
st.sidebar.header("🔄 Chargement du modèle")
model = None
model_name = None
with st.sidebar.spinner("Chargement depuis MLflow…"):
    try:
        EXP_NAME = "conso-electrique-xgboost"
        path = get_best_model_path_from_mlflow(
            EXP_NAME, metric_name="r2_val", maximize=True, artifact_subpath="model"
        )
        model = joblib.load(path)
        model_name = os.path.basename(path)
        st.sidebar.success(f"Modèle MLflow chargé: {model_name}")
    except Exception as e_ml:
        st.sidebar.error(f"MLflow KO: {e_ml}")
        # fallback
        local_models = glob.glob(os.path.join("models", "*.pkl"))
        if local_models:
            latest = max(local_models, key=os.path.getmtime)
            model = joblib.load(latest)
            model_name = os.path.basename(latest)
            st.sidebar.info(f"Fallback local: {model_name}")
        else:
            st.sidebar.error("Aucun modèle local trouvé non plus")

if model is None:
    st.error("Aucun modèle disponible, vérifiez les logs en sidebar.")
    st.stop()

# --- Préparation des données ---
def preparer_donnees(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time
    df['DateTime'] = df.apply(lambda r: datetime.combine(r['Date'], r['Heures']), axis=1)
    df = df.sort_values('DateTime').reset_index(drop=True)

    df['Weekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['mois'] = df['Date'].dt.month
    df['annee'] = df['Date'].dt.year
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    saison_mapping = {'Automne': 1, 'Hiver': 2, 'Printemps': 3, 'Été': 4}
    df['saison_num'] = df['Saison'].map(saison_mapping).fillna(0).astype(int)
    tempo_mapping = {'BLEU': 1, 'BLANC': 2, 'ROUGE': 3}
    df['tempo_num'] = df.get('Type de jour TEMPO', pd.Series(dtype=int)).map(tempo_mapping).fillna(0).astype(int)

    def encode_periode(t):
        if pd.isnull(t): return 0
        if t < dt_time(5): return 1
        if t < dt_time(12): return 2
        if t < dt_time(18): return 3
        return 4
    df['periode_jour_code'] = df['Heures'].apply(encode_periode)

    def get_holidays(year):
        fixed = [
            date(year,1,1), date(year,5,1), date(year,5,8),
            date(year,7,14), date(year,8,15),
            date(year,11,1), date(year,11,11), date(year,12,25)
        ]
        e = easter(year)
        movable = [e + pd.Timedelta(days=d) for d in (1,39,50)]
        return fixed + movable

    years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
    holidays = pd.to_datetime([day for y in years for day in get_holidays(y)])
    df['jour_ferie'] = df['Date'].dt.normalize().isin(holidays).astype(int)
    df['Heures_float'] = df['Heures'].apply(lambda t: t.hour + t.minute/60 if pd.notnull(t) else np.nan)

    for lag in (1,2,3,4):
        df[f'lag_{lag}'] = df['Consommation'].shift(lag)
    df = df.dropna().reset_index(drop=True)

    drop_cols = ['Date','Heures','Saison','Type de jour TEMPO','Jour','Prévision J','Prévision J-1']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    X = df.drop(columns=['Consommation','DateTime'])
    y = df['Consommation']
    idx = pd.DatetimeIndex(df['DateTime'])
    return X, y, idx

# --- Interface principale ---
st.title("Comparaison consommations réelles vs prédiction")

# Chargement test
DEFAULT_TEST = 'test_data.tsv'
if os.path.exists(DEFAULT_TEST):
    df_test = pd.read_csv(DEFAULT_TEST, sep='\t', encoding='latin1')
    st.info(f"Jeu de test chargé automatiquement ({DEFAULT_TEST})")
else:
    df_test = None
    up = st.file_uploader("Upload test (TSV)", type=['csv','tsv'])
    if up:
        df_test = pd.read_csv(up, sep='\t', encoding='latin1')

if df_test is not None:
    X_test, y_test, idx = preparer_donnees(df_test)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.subheader(f"Performance du modèle `{model_name}`")
    st.write(f"- RMSE : **{rmse:.2f}**")
    st.write(f"- R² : **{r2:.3f}**")

    df_plot = pd.DataFrame({
        "Réel": y_test.values,
        "Prédiction": preds
    }, index=idx)
    st.subheader("Courbe temps réel vs prédiction")
    st.line_chart(df_plot)

# Prédiction unique via API
st.sidebar.header("🔮 Prédiction unique via API")
dt_input = st.sidebar.text_input("Datetime ISO", "2025-05-15T12:00:00Z")
api_url = st.sidebar.text_input("API URL", "http://localhost:8001/predict/")
if st.sidebar.button("Prédire"):
    try:
        r = requests.get(api_url, params={'datetime_iso': dt_input}, timeout=5)
        r.raise_for_status()
        d = r.json()
        st.sidebar.success(f"{d['prediction']:.2f}")
        st.sidebar.json(d)
    except Exception as e:
        st.sidebar.error(f"Erreur API : {e}")

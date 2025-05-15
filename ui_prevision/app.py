import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime, date, time as dt_time
from dateutil.easter import easter
from sklearn.metrics import mean_squared_error, r2_score

# --- Prparation des données ---
def preparer_donnees(df: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DatetimeIndex):
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
    df['tempo_num'] = df.get('Type de jour TEMPO', pd.Series()).map(tempo_mapping).fillna(0).astype(int)

    def encode_periode(t):
        if pd.isnull(t): return 0
        if t < dt_time(5): return 1
        if t < dt_time(12): return 2
        if t < dt_time(18): return 3
        return 4
    df['periode_jour_code'] = df['Heures'].apply(encode_periode)

    def get_holidays(year):
        fixed = [date(year,1,1), date(year,5,1), date(year,5,8), date(year,7,14), date(year,8,15), date(year,11,1), date(year,11,11), date(year,12,25)]
        e = easter(year)
        movable = [e + pd.Timedelta(days=1), e + pd.Timedelta(days=39), e + pd.Timedelta(days=50)]
        return fixed + movable
    years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
    holidays = pd.to_datetime([d for y in years for d in get_holidays(y)])
    df['jour_ferie'] = df['Date'].dt.normalize().isin(holidays).astype(int)
    df['Heures_float'] = df['Heures'].apply(lambda t: t.hour + t.minute/60 if pd.notnull(t) else np.nan)

    for lag in [1,2,3,4]:
        df[f'lag_{lag}'] = df['Consommation'].shift(lag)
    df = df.dropna().reset_index(drop=True)

    drop_cols = ['Date','Heures','Saison','Type de jour TEMPO','Jour','Prévision J','Prévision J-1']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    X = df.drop(columns=['Consommation','DateTime'])
    y = df['Consommation']
    index = pd.DatetimeIndex(df['DateTime'])
    return X, y, index

# --- Streamlit UI ---
st.title("Comparaison de modèles XGBoost pour la consommation électrique")

# Chargement des modèles
model_paths = glob.glob(os.path.join('models', '*.pkl'))
models = {os.path.basename(p): joblib.load(p) for p in model_paths}

# Chargement du test automatique
DEFAULT_TEST = 'test_data.tsv'
if os.path.exists(DEFAULT_TEST):
    df_test = pd.read_csv(DEFAULT_TEST, sep='\t', encoding='latin1')
    st.info(f"Chargement automatique du test depuis {DEFAULT_TEST}")
else:
    df_test = None

# Uploader si nécessaire
if df_test is None:
    data_file = st.file_uploader("Upload du fichier de test (TSV)", type=['csv','tsv'])
    if data_file:
        df_test = pd.read_csv(data_file, sep='\t', encoding='latin1')

# Calcul et affichage direct de la comparaison
if models and df_test is not None:
    X_test, y_test, idx = preparer_donnees(df_test)
    preds_df = pd.DataFrame(index=idx)
    preds_df['Réel'] = y_test.values

    metrics = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        metrics.append({'model': name, 'RMSE': rmse, 'R2': r2})
        preds_df[name] = preds

    df_metrics = pd.DataFrame(metrics).set_index('model')
    st.subheader("Métriques de performance")
    st.table(df_metrics)

    # Affichage du meilleur modèle selon RMSE
    best_rmse = df_metrics['RMSE'].idxmin()
    best_values = df_metrics.loc[best_rmse]
    st.markdown(f"**Meilleur modèle :** {best_rmse} (RMSE={best_values['RMSE']:.2f}, R2={best_values['R2']:.3f})")

    st.subheader("Courbes de consommation : Réel vs Modèles")
    st.line_chart(preds_df)

# Prédiction unique via API
st.sidebar.header("Prédiction unique via API")

dt_input = st.sidebar.text_input("Datetime ISO", "2025-05-15T12:00:00Z")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000/predict/")
if st.sidebar.button("Prédire via API"):
    import requests
    try:
        response = requests.get(api_url, params={'datetime_iso': dt_input}, timeout=5)
        response.raise_for_status()
        data = response.json()
        st.sidebar.success(f"Prédiction API: {data.get('prediction'):.2f}")
        st.sidebar.write(data)
    except Exception as e:
        st.sidebar.error(f"Erreur API: {e}")

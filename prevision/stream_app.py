import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import date, time
from dateutil.easter import easter
from sklearn.metrics import mean_squared_error, r2_score

# --- Préparation des données (identique à l'API) ---
def preparer_donnees(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time
    df['Weekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['mois'] = df['Date'].dt.month
    df['annee'] = df['Date'].dt.year
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    saison_mapping = {'Automne': 1, 'Hiver': 2, 'Printemps': 3, 'Été': 4}
    df['saison_num'] = df['Saison'].map(saison_mapping).fillna(0).astype(int)
    if 'Type de jour TEMPO' in df.columns:
        tempo_mapping = {'BLEU': 1, 'BLANC': 2, 'ROUGE': 3}
        df['tempo_num'] = df['Type de jour TEMPO'].map(tempo_mapping).fillna(0).astype(int)
    else:
        df['tempo_num'] = 0
    def encode_periode(t):
        if pd.isnull(t): return 0
        if t < time(5): return 1
        if t < time(12): return 2
        if t < time(18): return 3
        return 4
    df['periode_jour_code'] = df['Heures'].apply(encode_periode)
    def get_holidays(year):
        fixed = [date(year,1,1), date(year,5,1), date(year,5,8), date(year,7,14), date(year,8,15), date(year,11,1), date(year,11,11), date(year,12,25)]
        e = easter(year)
        movable = [e + pd.Timedelta(days=1), e + pd.Timedelta(days=39), e + pd.Timedelta(days=50)]
        return fixed + movable
    years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
    holidays = pd.to_datetime([h for y in years for h in get_holidays(y)])
    df['jour_ferie'] = df['Date'].dt.normalize().isin(holidays).astype(int)
    df['Heures_float'] = df['Heures'].apply(lambda t: t.hour + t.minute/60 if pd.notnull(t) else np.nan)
    df = df.sort_values(['Date','Heures_float']).reset_index(drop=True)
    for lag in [1,2,3,4]: df[f'lag_{lag}'] = df['Consommation'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    drop_cols = ['Date','Heures','Saison','Type de jour TEMPO','Jour','Prévision J','Prévision J-1']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    X = df.drop(columns=['Consommation'])
    y = df['Consommation']
    return X, y

# --- Streamlit UI ---
st.title("Comparaison de modèles XGBoost pour la consommation électrique")

# Chargement automatique des modèles depuis le dossier 'models'
model_paths = glob.glob(os.path.join('models', '*.pkl'))
models = {os.path.basename(p): joblib.load(p) for p in model_paths}

# Chargement des données de test
data_file = st.file_uploader("Upload du fichier de test (TSV)", type=['csv','tsv'])

if models and data_file:
    df_test = pd.read_csv(data_file, sep='\t', encoding='latin1')
    X_test, y_test = preparer_donnees(df_test)
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        results.append({
            'model': name,
            'RMSE': mean_squared_error(y_test, preds, squared=False),
            'R2': r2_score(y_test, preds)
        })
    df_results = pd.DataFrame(results).set_index('model')
    st.subheader("Métriques de performance")
    st.table(df_results)
    st.subheader("Comparaison des RMSE")
    st.bar_chart(df_results['RMSE'])
    st.subheader("Comparaison des R²")
    st.bar_chart(df_results['R2'])

# Prédiction à la date&heure unique via API
st.sidebar.header("Prédiction unique via API")
dt_input = st.sidebar.text_input("Datetime ISO", "2025-05-15T12:00:00Z")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000/predict/")
if st.sidebar.button("Prédire via API"):
    import requests
    try:
        params = {'datetime_iso': dt_input}
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        pred_val = data.get('prediction')
        st.sidebar.success(f"Prédiction API: {pred_val:.2f}")
        st.sidebar.write(data)
    except Exception as e:
        st.sidebar.error(f"Erreur API: {e}")

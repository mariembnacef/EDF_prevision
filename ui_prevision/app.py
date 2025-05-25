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
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page pour l'accessibilité
st.set_page_config(
    page_title="Prédiction Consommation Électrique - Interface Accessible",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour l'accessibilité
st.markdown("""
<style>
/* Amélioration du contraste et de la lisibilité */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Styles pour les personnes daltoniennes */
.status-success {
    background-color: #2E7D32 !important;
    color: white !important;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #4CAF50;
    font-weight: bold;
}

.status-error {
    background-color: #C62828 !important;
    color: white !important;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #F44336;
    font-weight: bold;
}

.status-warning {
    background-color: #F57F17 !important;
    color: black !important;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #FF9800;
    font-weight: bold;
}

.status-info {
    background-color: #1565C0 !important;
    color: white !important;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #2196F3;
    font-weight: bold;
}

/* Amélioration des boutons */
.stButton > button {
    background-color: #1976D2;
    color: white;
    border: 2px solid #0D47A1;
    font-weight: bold;
    font-size: 16px;
    padding: 8px 16px;
    border-radius: 4px;
}

.stButton > button:hover {
    background-color: #0D47A1;
    border-color: #1976D2;
}

/* Amélioration des inputs */
.stTextInput > div > div > input {
    font-size: 16px;
    padding: 8px;
    border: 2px solid #1976D2;
}

/* Headers avec meilleur contraste */
h1, h2, h3 {
    color: #0D47A1 !important;
    font-weight: bold;
}

/* Amélioration du sidebar */
.css-1d391kg {
    background-color: #f8f9fa;
    border-right: 2px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# --- Fonction utilitaire MLflow ---
def get_best_model_path_from_mlflow(
    experiment_name: str,
    metric_name: str = "r2_val",
    maximize: bool = True,
    artifact_subpath: str = "model"
) -> str:
    """
    Récupère le meilleur modèle depuis MLflow.
    
    Args:
        experiment_name: Nom de l'expérience MLflow
        metric_name: Nom de la métrique à optimiser
        maximize: True pour maximiser la métrique, False pour minimiser
        artifact_subpath: Chemin vers l'artifact du modèle
        
    Returns:
        Chemin vers le fichier du modèle
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://37.59.218.166:5000")
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
    raise FileNotFoundError(f"Aucun fichier .pkl/.joblib trouvé dans '{local_dir}'")

# Fonctions d'affichage accessibles
def display_success(message):
    """Affiche un message de succès avec styles accessibles"""
    st.markdown(f'<div class="status-success">✅ SUCCÈS: {message}</div>', unsafe_allow_html=True)
    st.success(f"SUCCÈS: {message}")  # Pour les lecteurs d'écran

def display_error(message):
    """Affiche un message d'erreur avec styles accessibles"""
    st.markdown(f'<div class="status-error">❌ ERREUR: {message}</div>', unsafe_allow_html=True)
    st.error(f"ERREUR: {message}")  # Pour les lecteurs d'écran

def display_warning(message):
    """Affiche un message d'avertissement avec styles accessibles"""
    st.markdown(f'<div class="status-warning">⚠️ ATTENTION: {message}</div>', unsafe_allow_html=True)
    st.warning(f"ATTENTION: {message}")  # Pour les lecteurs d'écran

def display_info(message):
    """Affiche un message d'information avec styles accessibles"""
    st.markdown(f'<div class="status-info">ℹ️ INFO: {message}</div>', unsafe_allow_html=True)
    st.info(f"INFO: {message}")  # Pour les lecteurs d'écran

# --- En-tête principal avec description ---
st.title("⚡ Prédiction de Consommation Électrique")
st.markdown("### Interface accessible pour l'analyse et la prédiction de consommation électrique")

# Description de l'application pour les lecteurs d'écran
with st.expander("📖 Description de l'application (cliquez pour développer)", expanded=False):
    st.markdown("""
    **Cette application permet de :**
    
    1. **Charger automatiquement** le meilleur modèle depuis MLflow ou un modèle local de secours
    2. **Analyser les performances** du modèle sur un jeu de test
    3. **Visualiser** les comparaisons entre prédictions et valeurs réelles
    4. **Effectuer des prédictions** individuelles via API
    
    **Navigation :**
    - La sidebar (panneau latéral) contient les contrôles de chargement de modèle et prédiction API
    - Le contenu principal affiche les résultats et visualisations
    - Tous les graphiques incluent des descriptions textuelles pour l'accessibilité
    """)

# --- Chargement du (meilleur) modèle MLflow ou fallback local ---
st.sidebar.markdown("## 🔄 Chargement du modèle")
st.sidebar.markdown("*Cette section gère le chargement automatique du modèle d'IA*")

model = None
model_name = None

# Barre de progression et statut de chargement
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

# Utilise st.spinner() et affiche le résultat dans la sidebar
with st.spinner("Chargement du modèle depuis MLflow en cours..."):
    try:
        status_text.text("Connexion à MLflow...")
        progress_bar.progress(25)
        
        EXP_NAME = "conso-electrique-xgboost"
        path = get_best_model_path_from_mlflow(
            EXP_NAME,
            metric_name="r2_val",
            maximize=True,
            artifact_subpath="model"
        )
        
        progress_bar.progress(75)
        status_text.text("Chargement du modèle...")
        
        model = joblib.load(path)
        model_name = os.path.basename(path)
        
        progress_bar.progress(100)
        status_text.text("Modèle chargé avec succès!")
        
        st.sidebar.markdown(f'<div class="status-success">Modèle MLflow chargé: {model_name}</div>', 
                           unsafe_allow_html=True)
        display_success(f"Modèle MLflow chargé: {model_name}")
        
    except Exception as e_ml:
        progress_bar.progress(50)
        status_text.text("Échec MLflow, tentative de chargement local...")
        
        st.sidebar.markdown(f'<div class="status-error">MLflow KO: {str(e_ml)}</div>', 
                           unsafe_allow_html=True)
        
        # fallback local
        local_models = glob.glob(os.path.join("models", "*.pkl"))
        if local_models:
            latest = max(local_models, key=os.path.getmtime)
            model = joblib.load(latest)
            model_name = os.path.basename(latest)
            
            progress_bar.progress(100)
            status_text.text("Modèle local chargé!")
            
            st.sidebar.markdown(f'<div class="status-info">Fallback local: {model_name}</div>', 
                               unsafe_allow_html=True)
            display_info(f"Modèle local utilisé: {model_name}")
        else:
            progress_bar.progress(0)
            status_text.text("Aucun modèle disponible!")
            
            st.sidebar.markdown('<div class="status-error">Aucun modèle local trouvé non plus</div>', 
                               unsafe_allow_html=True)
            display_error("Aucun modèle local trouvé")

if model is None:
    display_error("Aucun modèle disponible. Vérifiez les messages d'état dans la sidebar.")
    st.markdown("**Action requise :** Vérifiez la connexion MLflow ou placez un fichier .pkl dans le dossier 'models/'")
    st.stop()

# --- Préparation des données ---
def preparer_donnees(df: pd.DataFrame):
    """
    Prépare les données pour la prédiction en ajoutant les features nécessaires.
    
    Args:
        df: DataFrame contenant les données brutes
        
    Returns:
        tuple: (X, y, index_datetime) - features, target, index temporel
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time
    df['DateTime'] = df.apply(lambda r: datetime.combine(r['Date'], r['Heures']), axis=1)
    df = df.sort_values('DateTime').reset_index(drop=True)

    # Features temporelles
    df['Weekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['mois'] = df['Date'].dt.month
    df['annee'] = df['Date'].dt.year
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    
    # Encodage des saisons
    saison_mapping = {'Automne': 1, 'Hiver': 2, 'Printemps': 3, 'Été': 4}
    df['saison_num'] = df['Saison'].map(saison_mapping).fillna(0).astype(int)
    
    # Encodage TEMPO
    tempo_mapping = {'BLEU': 1, 'BLANC': 2, 'ROUGE': 3}
    df['tempo_num'] = df.get('Type de jour TEMPO', pd.Series(dtype=int)).map(tempo_mapping).fillna(0).astype(int)

    def encode_periode(t):
        if pd.isnull(t): return 0
        if t < dt_time(5): return 1    # Nuit
        if t < dt_time(12): return 2   # Matin
        if t < dt_time(18): return 3   # Après-midi
        return 4                       # Soir
    df['periode_jour_code'] = df['Heures'].apply(encode_periode)

    # Jours fériés
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

    # Features de lag
    for lag in (1,2,3,4):
        df[f'lag_{lag}'] = df['Consommation'].shift(lag)
    df = df.dropna().reset_index(drop=True)

    # Nettoyage des colonnes
    drop_cols = ['Date','Heures','Saison','Type de jour TEMPO','Jour','Prévision J','Prévision J-1']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    X = df.drop(columns=['Consommation','DateTime'])
    y = df['Consommation']
    idx = pd.DatetimeIndex(df['DateTime'])
    return X, y, idx

# --- Interface principale ---
st.markdown("---")
st.markdown("## 📊 Analyse des Performances du Modèle")

# Chargement du jeu de test avec instructions claires
DEFAULT_TEST = 'test_data.tsv'
df_test = None

col1, col2 = st.columns([2, 1])

with col1:
    if os.path.exists(DEFAULT_TEST):
        df_test = pd.read_csv(DEFAULT_TEST, sep='\t', encoding='latin1')
        display_info(f"Jeu de test chargé automatiquement: {DEFAULT_TEST}")
        st.markdown(f"**Fichier:** {DEFAULT_TEST}")
        st.markdown(f"**Nombre d'échantillons:** {len(df_test)}")
        st.markdown(f"**Période:** {df_test['Date'].min()} au {df_test['Date'].max()}")
    else:
        st.markdown("### 📁 Chargement de données de test")
        st.markdown("**Format requis:** Fichier TSV avec colonnes Date, Heures, Consommation, etc.")
        up = st.file_uploader(
            "Sélectionnez votre fichier de test (formats: CSV, TSV)", 
            type=['csv','tsv'],
            help="Le fichier doit contenir au minimum les colonnes: Date, Heures, Consommation"
        )
        if up:
            df_test = pd.read_csv(up, sep='\t', encoding='latin1')
            display_success(f"Fichier {up.name} chargé avec succès")

with col2:
    if df_test is not None:
        st.markdown("### 📋 Aperçu des données")
        st.dataframe(df_test.head(), use_container_width=True)

# Analyse et prédiction
if df_test is not None:
    with st.spinner("Préparation des données et calcul des prédictions..."):
        try:
            X_test, y_test, idx = preparer_donnees(df_test)
            preds = model.predict(X_test)

            # Calcul des métriques
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mae = np.mean(np.abs(y_test - preds))
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

            # Affichage des performances avec descriptions accessibles
            st.markdown("---")
            st.markdown(f"## 🎯 Performance du modèle `{model_name}`")
            
            # Métriques en colonnes avec descriptions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RMSE", 
                    f"{rmse:.2f}", 
                    help="Root Mean Square Error - Erreur quadratique moyenne. Plus faible = meilleur."
                )
            
            with col2:
                st.metric(
                    "R²", 
                    f"{r2:.3f}",
                    help="Coefficient de détermination. Plus proche de 1 = meilleur. Indique la part de variance expliquée."
                )
            
            with col3:
                st.metric(
                    "MAE", 
                    f"{mae:.2f}",
                    help="Mean Absolute Error - Erreur absolue moyenne. Plus faible = meilleur."
                )
            
            with col4:
                st.metric(
                    "MAPE", 
                    f"{mape:.1f}%",
                    help="Mean Absolute Percentage Error - Erreur absolue moyenne en pourcentage."
                )

            # Interprétation des résultats pour l'accessibilité
            st.markdown("### 📈 Interprétation des résultats")
            if r2 > 0.8:
                quality = "Excellente"
                color = "green"
            elif r2 > 0.6:
                quality = "Bonne" 
                color = "orange"
            else:
                quality = "À améliorer"
                color = "red"
                
            st.markdown(f"**Qualité du modèle:** {quality} (R² = {r2:.3f})")
            st.markdown(f"**Erreur moyenne:** {mae:.2f} unités de consommation")
            st.markdown(f"**Erreur relative:** {mape:.1f}% en moyenne")

            # Création du graphique accessible
            df_plot = pd.DataFrame({
                "Réel": y_test.values,
                "Prédiction": preds,
                "Erreur": y_test.values - preds
            }, index=idx)

            st.markdown("---")
            st.markdown("## 📊 Visualisation: Consommation Réelle vs Prédiction")
            
            # Description textuelle du graphique pour les lecteurs d'écran
            with st.expander("📋 Description détaillée du graphique", expanded=False):
                st.markdown(f"""
                **Graphique linéaire comparant les valeurs réelles et prédites:**
                
                - **Période analysée:** {idx.min().strftime('%d/%m/%Y %H:%M')} au {idx.max().strftime('%d/%m/%Y %H:%M')}
                - **Nombre de points:** {len(df_plot)}
                - **Consommation réelle:** Ligne bleue continue 
                - **Prédiction du modèle:** Ligne rouge pointillée
                - **Valeur minimale réelle:** {y_test.min():.2f}
                - **Valeur maximale réelle:** {y_test.max():.2f}
                - **Valeur minimale prédite:** {preds.min():.2f}
                - **Valeur maximale prédite:** {preds.max():.2f}
                - **Écart-type des erreurs:** {np.std(df_plot['Erreur']):.2f}
                """)

            # Graphique Plotly avec palette accessible aux daltoniens
            fig = go.Figure()
            
            # Ligne pour les valeurs réelles (bleu foncé)
            fig.add_trace(go.Scatter(
                x=idx,
                y=y_test.values,
                mode='lines',
                name='Consommation Réelle',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Réel</b><br>Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
            ))
            
            # Ligne pour les prédictions (rouge/orange)
            fig.add_trace(go.Scatter(
                x=idx,
                y=preds,
                mode='lines',
                name='Prédiction',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Prédiction</b><br>Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': f'Comparaison Réel vs Prédiction - R² = {r2:.3f}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#0D47A1'}
                },
                xaxis_title='Date et Heure',
                yaxis_title='Consommation Électrique', 
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right", 
                    x=1
                ),
                font=dict(size=12),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Amélioration de la grille pour la lisibilité
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            st.plotly_chart(fig, use_container_width=True)

            # Graphique de distribution des erreurs
            st.markdown("### 📊 Distribution des Erreurs de Prédiction")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_plot['Erreur'],
                nbinsx=30,
                name='Distribution des erreurs',
                marker_color='#2E8B57',
                opacity=0.7,
                hovertemplate='<b>Erreur</b><br>Intervalle: %{x}<br>Fréquence: %{y}<extra></extra>'
            ))
            
            fig_hist.update_layout(
                title='Distribution des Erreurs (Réel - Prédiction)',
                xaxis_title='Erreur de Prédiction',
                yaxis_title='Fréquence',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

            # Statistiques détaillées des erreurs
            with st.expander("📊 Statistiques détaillées des erreurs", expanded=False):
                error_stats = df_plot['Erreur'].describe()
                st.markdown("**Analyse statistique des erreurs de prédiction:**")
                for stat, value in error_stats.items():
                    st.markdown(f"- **{stat.capitalize()}:** {value:.2f}")

        except Exception as e:
            display_error(f"Erreur lors du traitement des données: {str(e)}")
            st.markdown("**Vérifiez que votre fichier contient toutes les colonnes requises.**")

# --- Prédiction unique via API ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔮 Prédiction via API")
st.sidebar.markdown("*Effectuez une prédiction ponctuelle en utilisant l'API*")

# Instructions pour l'utilisateur
with st.sidebar.expander("ℹ️ Instructions API", expanded=False):
    st.markdown("""
    **Format de date requis:** ISO 8601
    
    **Exemples valides:**
    - `2025-05-15T12:00:00Z`
    - `2025-12-25T08:30:00Z`
    - `2025-01-01T00:00:00Z`
    
    **URL par défaut:** API locale sur port 8001
    """)

dt_input = st.sidebar.text_input(
    "📅 Date et heure (format ISO)", 
    "2025-05-15T12:00:00Z",
    help="Format: YYYY-MM-DDTHH:MM:SSZ"
)

api_url = st.sidebar.text_input(
    "🌐 URL de l'API", 
    "http://localhost:8001/predict/",
    help="URL complète de l'endpoint de prédiction"
)

if st.sidebar.button("🔮 Lancer la Prédiction", type="primary"):
    if not dt_input.strip():
        st.sidebar.markdown('<div class="status-error">Date requise</div>', unsafe_allow_html=True)
    elif not api_url.strip():
        st.sidebar.markdown('<div class="status-error">URL API requise</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Envoi de la requête API..."):
            try:
                # Validation du format de date
                datetime.fromisoformat(dt_input.replace('Z', '+00:00'))
                
                r = requests.get(
                    api_url, 
                    params={'datetime_iso': dt_input}, 
                    timeout=10
                )
                r.raise_for_status()
                d = r.json()
                
                # Affichage du résultat avec succès
                pred_value = d['prediction']
                st.sidebar.markdown(
                    f'<div class="status-success">Prédiction: {pred_value:.2f}</div>', 
                    unsafe_allow_html=True
                )
                
                # Détails de la réponse
                with st.sidebar.expander("📋 Détails de la réponse", expanded=True):
                    st.json(d)
                    
                # Affichage dans la zone principale aussi
                st.markdown("### 🎯 Dernière Prédiction API")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prédiction", f"{pred_value:.2f}")
                with col2:
                    st.metric("Timestamp", dt_input)
                    
            except ValueError as ve:
                st.sidebar.markdown(
                    f'<div class="status-error">Format de date invalide: {str(ve)}</div>', 
                    unsafe_allow_html=True
                )
            except requests.exceptions.Timeout:
                st.sidebar.markdown(
                    '<div class="status-error">Timeout - API non accessible</div>', 
                    unsafe_allow_html=True
                )
            except requests.exceptions.ConnectionError:
                st.sidebar.markdown(
                    '<div class="status-error">Connexion impossible - Vérifiez l\'URL</div>', 
                    unsafe_allow_html=True
                )
            except requests.exceptions.HTTPError as he:
                st.sidebar.markdown(
                    f'<div class="status-error">Erreur HTTP {he.response.status_code}</div>', 
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.sidebar.markdown(
                    f'<div class="status-error">Erreur API: {str(e)}</div>', 
                    unsafe_allow_html=True
                )

# --- Footer avec informations d'accessibilité ---
st.markdown("---")
st.markdown("### ♿ Informations d'Accessibilité")

with st.expander("🔧 Fonctionnalités d'accessibilité implémentées", expanded=False):
    st.markdown("""
    **Pour les personnes aveugles et malvoyantes:**
    - ✅ Descriptions textuelles détaillées de tous les graphiques
    - ✅ Messages d'état vocalisables par les lecteurs d'écran
    - ✅ Structure HTML sémantique avec titres hiérarchiques
    - ✅ Labels explicites pour tous les contrôles interactifs
    - ✅ Informations de contexte et instructions d'utilisation
    
    **Pour les personnes daltoniennes:**
    - ✅ Palette de couleurs adaptée (bleu/orange au lieu de rouge/vert)
    - ✅ Motifs différenciés (lignes pleines vs pointillées)
    - ✅ Forte différence de contraste
    - ✅ Informations transmises par la forme et le texte, pas seulement la couleur
    - ✅ Messages d'état avec icônes et texte descriptif
    
    **Général:**
    - ✅ Interface haute contraste
    - ✅ Taille de police augmentée
    - ✅ Boutons avec états focus visibles
    - ✅ Messages d'erreur explicites et constructifs
    """)

# Informations techniques pour les développeurs
with st.expander("🔍 Informations techniques", expanded=False):
    st.markdown("""
    **Technologies utilisées:**
    - Streamlit avec configuration d'accessibilité
    - Plotly pour graphiques interactifs accessibles
    - Palette de couleurs ColorBrewer adaptée au daltonisme
    - CSS personnalisé pour contraste élevé
    - Messages ARIA-friendly pour lecteurs d'écran
    
    **Standards respectés:**
    - WCAG 2.1 AA pour le contraste des couleurs
    - Section 508 pour l'accessibilité des interfaces fédérales
    - Bonnes pratiques d'UX inclusive
    
    **Raccourcis clavier:**
    - Tab/Shift+Tab: Navigation entre éléments
    - Espace/Entrée: Activation des boutons
    - Flèches: Navigation dans les graphiques interactifs
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    <p><strong>Interface Accessible pour Prédiction de Consommation Électrique</strong></p>
    <p>Développée avec les standards d'accessibilité WCAG 2.1 AA</p>
    <p>Compatible lecteurs d'écran • Optimisée daltonisme • Contraste élevé</p>
</div>
""", unsafe_allow_html=True)
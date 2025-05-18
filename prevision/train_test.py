#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'entraînement et d'évaluation de modèle pour prédire la consommation électrique
Version optimisée pour améliorer la vitesse d'exécution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
import joblib
from dateutil.easter import easter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import io
from contextlib import redirect_stdout
import time
from mlflow.models.signature import infer_signature


def preparer_donnees(chemin_fichier):
    """
    Charge et prépare les données pour la modélisation
    """
    mlflow.log_param("data_source", chemin_fichier)
    start_time = time.time()

    df = pd.read_csv(chemin_fichier, sep="\t", encoding="latin1")
    mlflow.log_param("n_rows_initial", len(df))

    # Conversion dates et heures
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Heures'] = pd.to_datetime(df['Heures'], format='%H:%M', errors='coerce').dt.time

    # Variables temporelles
    df['mois'] = df['Date'].dt.month
    df['annee'] = df['Date'].dt.year
    df['jour_semaine'] = df['Date'].dt.weekday + 1
    saison_mapping = {'Automne':1,'Hiver':2,'Printemps':3,'Été':4}
    df['saison_num'] = df['Saison'].map(saison_mapping)

    # TEMPO
    if 'Type de jour TEMPO' in df.columns:
        df['tempo_num'] = df['Type de jour TEMPO'].map({'BLEU':1,'BLANC':2,'ROUGE':3})
        mlflow.log_param("tempo_available", True)
    else:
        mlflow.log_param("tempo_available", False)

    # Période jour
    def encoder_periode_jour(t):
        if pd.isnull(t): return None
        if dt.time(0)<=t<dt.time(5): return 1
        if dt.time(5)<=t<dt.time(12): return 2
        if dt.time(12)<=t<dt.time(18): return 3
        return 4
    df['periode_jour_code'] = df['Heures'].apply(encoder_periode_jour)

    # Jours fériés
    def get_french_holidays(years):
        hol=[]
        for y in years:
            fixed=[dt.date(y,1,1),dt.date(y,5,1),dt.date(y,5,8),dt.date(y,7,14),
                   dt.date(y,8,15),dt.date(y,11,1),dt.date(y,11,11),dt.date(y,12,25)]
            e=easter(y)
            mov=[e+dt.timedelta(days=d) for d in (1,39,50)]
            hol.extend(fixed+mov)
        return pd.to_datetime(hol)
    years=range(df['Date'].dt.year.min(), df['Date'].dt.year.max()+1)
    all_holidays=get_french_holidays(years)
    df['jour_ferie']=df['Date'].dt.normalize().isin(all_holidays)

    # Heures float
    df['Heures']=pd.to_datetime(df['Heures'],format='%H:%M:%S',errors='coerce')
    df['Heures_float']=df['Heures'].dt.hour+df['Heures'].dt.minute/60

    # Lags
    df=df.sort_values(['Date','Heures_float']).reset_index(drop=True)
    for lag in range(1,5): df[f'lag_{lag}']=df['Consommation'].shift(lag)
    df=df.dropna().reset_index(drop=True)

    # Sélection colonnes
    drop_cols=[c for c in ['Type de jour TEMPO','Date','Heures','Prévision J','Prévision J-1','Jour','Saison'] if c in df.columns]
    df_model=df.drop(columns=drop_cols).dropna()

    X=df_model.drop(columns=['Consommation'])
    y=df_model['Consommation']

    # Logging MLflow
    mlflow.log_param("n_rows_final", len(df_model))
    mlflow.log_param("min_date", df['Date'].min().strftime('%Y-%m-%d'))
    mlflow.log_param("max_date", df['Date'].max().strftime('%Y-%m-%d'))
    mlflow.log_param("n_features", X.shape[1])
    stats=pd.DataFrame({'min':[y.min()],'max':[y.max()],'mean':[y.mean()],'std':[y.std()]},index=['Consommation'])
    buf=io.StringIO(); stats.to_csv(buf)
    mlflow.log_text(buf.getvalue(),"data_stats.csv")
    mlflow.log_param("features",X.columns.tolist())
    mlflow.log_metric("data_prep_time_seconds", time.time()-start_time)

    print(f"Colonnes: {X.columns.tolist()}")
    return X,y,df_model


def entrainer_modele(X,y,verbose=True,mode_rapide=True):
    """Entraînement XGBoost optimisé"""
    start=time.time()
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,shuffle=False)
    mlflow.log_param("mode_rapide",mode_rapide)

    if mode_rapide:
        params={'n_estimators':300,'learning_rate':0.1,'max_depth':6,'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':1.0}
        for k,v in params.items(): mlflow.log_param(f"quick_{k}",v)
        model=XGBRegressor(objective='reg:squarederror',random_state=42,**params)
        model.fit(X_train,y_train)
    else:
        dist={'n_estimators':[100,300,500],'learning_rate':[0.05,0.1,0.2],'max_depth':[4,6,8],'subsample':[0.8,1.0],'colsample_bytree':[0.8,1.0],'reg_alpha':[0,0.1],'reg_lambda':[0.5,1]}
        mlflow.log_param("random_iters",10)
        rs=RandomizedSearchCV(XGBRegressor(objective='reg:squarederror',random_state=42),dist,n_iter=10,scoring='neg_root_mean_squared_error',cv=2,verbose=0,n_jobs=-1)
        rs.fit(X_train,y_train)
        model=rs.best_estimator_
        for k,v in rs.best_params_.items(): mlflow.log_param(f"best_{k}",v)

    # Importance features
    fi=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,8)); fi.plot(kind='barh'); plt.tight_layout(); plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    pd.DataFrame({'feature':fi.index,'importance':fi.values}).to_csv('feature_importances.csv',index=False)
    mlflow.log_artifact('feature_importances.csv')
    mlflow.log_metric("train_time_seconds", time.time()-start)
    return model,X_train,X_val,y_train,y_val


def evaluer_modele(model,X_train,X_val,y_train,y_val):
    y_train_pred=model.predict(X_train)
    y_val_pred=model.predict(X_val)
    rmse_train=np.sqrt(mean_squared_error(y_train,y_train_pred)); r2_train=r2_score(y_train,y_train_pred)
    rmse_val=np.sqrt(mean_squared_error(y_val,y_val_pred)); r2_val=r2_score(y_val,y_val_pred)
    diff=r2_train-r2_val
    mlflow.log_metric("rmse_train",rmse_train); mlflow.log_metric("r2_train",r2_train)
    mlflow.log_metric("rmse_val",rmse_val); mlflow.log_metric("r2_val",r2_val)
    mlflow.log_metric("r2_diff",diff); mlflow.log_param("overfitting", diff>0.05)

    # Artéfacts prédiction
    preds=pd.DataFrame({'actual':y_val,'pred':y_val_pred,'error':y_val-y_val_pred})
    preds.to_csv('validation_predictions.csv',index=False)
    mlflow.log_artifact('validation_predictions.csv')

    # Graphiques
    plt.figure(figsize=(10,8)); plt.scatter(y_val,y_val_pred,alpha=0.5); plt.plot([y_val.min(),y_val.max()],[y_val.min(),y_val.max()],'r--'); plt.savefig('pred_vs_actual.png'); mlflow.log_artifact('pred_vs_actual.png')
    plt.figure(figsize=(10,8)); sns.histplot(y_val-y_val_pred,kde=True); plt.savefig('error_dist.png'); mlflow.log_artifact('error_dist.png')

    print(f"Train R2: {r2_train:.4f}, RMSE: {rmse_train:.2f}")
    print(f"Val   R2: {r2_val:.4f}, RMSE: {rmse_val:.2f}")
    return {'rmse_val':rmse_val,'r2_val':r2_val,'overfitting': diff>0.05}


def executer_pipeline_complete(chemin_fichier,model_path=None,verbose=True,mode_rapide=True):
    with mlflow.start_run(nested=True) as run:
        mlflow.log_param("module","train_test")
        mlflow.log_param("file_path", chemin_fichier)
        mlflow.log_param("mode_rapide",mode_rapide)
        X,y,_=preparer_donnees(chemin_fichier)
        model,Xt,Xv,yt,yv=entrainer_modele(X,y,verbose,mode_rapide)
        res=evaluer_modele(model,Xt,Xv,yt,yv)
        if model_path:
            joblib.dump(model, model_path)
            mlflow.log_param("model_path", model_path)
        sig=infer_signature(Xt, model.predict(Xt))
        ie=Xt.head(5)
        mlflow.sklearn.log_model(model,"model",signature=sig,input_example=ie)
    return model,res


def sauvegarder_modele(model,nom_fichier_base="models/xgboost_conso_best_model"):
    timestamp=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier=os.path.dirname(nom_fichier_base)
    base=os.path.basename(nom_fichier_base)
    chemin=os.path.join(dossier,f"{base}_{timestamp}.pkl")
    os.makedirs(dossier,exist_ok=True)
    joblib.dump(model, chemin)
    mlflow.log_param("model_local_path", chemin)
    mlflow.log_artifact(chemin, "model_joblib")
    return chemin


def visualiser_resultats(y_test,y_pred,features=None,prefix=""):
    plt.figure(figsize=(10,8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Réel'); plt.ylabel('Prédit'); plt.title('Prédictions vs Réels')
    plt.savefig(f"{prefix}preds_vs_act.png"); mlflow.log_artifact(f"{prefix}preds_vs_act.png")
    if features is not None:
        plt.figure(figsize=(10,8)); features.plot(kind='barh'); plt.title('Importance'); plt.savefig(f"{prefix}feat_imp.png"); mlflow.log_artifact(f"{prefix}feat_imp.png")

if __name__ == "__main__":
    print("Ce module contient des fonctions pour entraîner et évaluer un modèle XGBoost.")
    print("Pour utiliser la pipeline complète, exécutez main.py")

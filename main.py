
# main.py - Pipeline complet de prévision de consommation électrique

from src.explore_data_modulaire import pipeline_exploration
from src.encoding_data_modulaire import pipeline_encodage
from src.model1_KNN_modulaire import pipeline_knn
from src.model2_Randomforest_modulaire import pipeline_rf
from src.model3_xgboost_feries import pipeline_xgb_feries
from src.model3_xgboost import pipeline_xgboost
from src.compare_models import pipeline_comparaison
from src.compare_predictions import comparer_predictions
def pipeline_global():
    print("🔍 Étape 1 : Exploration des données")
    pipeline_exploration()
    
    print("🧼 Étape 2 : Encodage des variables")
    pipeline_encodage()
    
    print("🤖 Étape 3 : Entraînement du modèle KNN")
    pipeline_knn()
    
    print("🌲 Étape 4 : Entraînement du modèle Random Forest")
    pipeline_rf()
    
    
    print("🤖 Étape 5 : Entraînement du modèle XGBOOST")
    pipeline_xgboost()
    
    print("🤖 Étape 6 : Entraînement du modèle XGBOOST JOUR FERIE")
    pipeline_xgb_feries()
    print("📊 Étape 7 : Comparaison graphique des scores")
    pipeline_comparaison()
    print("📈 Étape 8 : Comparaison graphique des prédictions")
    comparer_predictions()
    print("✅ Pipeline complet exécuté avec succès.")
    
    print("✅ Pipeline complet exécuté avec succès.")

if __name__ == "__main__":
    pipeline_global()


from src.explore_data_modulaire import pipeline_exploration
from src.encoding_data_modulaire import pipeline_encodage
from src.model1_KNN_modulaire import pipeline_knn, evaluer_modele as eval_knn, charger_donnees as charger_knn, preparation_donnees as prep_knn, entrainer_modele_knn
from src.model2_Randomforest_modulaire import pipeline_rf, evaluer_modele as eval_rf, charger_donnees as charger_rf, preparation_donnees as prep_rf, entrainer_modele_rf
from src.record_performance import enregistrer_performances_csv
from src.compare_models import pipeline_comparaison
from src.compare_predictions import comparer_predictions

def pipeline_global():
    print("🔍 Étape 1 : Exploration des données")
    pipeline_exploration()

    print("🧼 Étape 2 : Encodage des variables")
    pipeline_encodage()

    print("🤖 Étape 3 : Entraînement du modèle KNN")
    df_knn = charger_knn()
    if df_knn is not None:
        X_train, X_test, y_train, y_test = prep_knn(df_knn)
        model_knn, scaler_knn = entrainer_modele_knn(X_train, y_train)
        mae_knn, rmse_knn, r2_knn = eval_knn(model_knn, scaler_knn, X_test, y_test)
        enregistrer_performances_csv("KNN", mae_knn, rmse_knn, r2_knn)

    print("🌲 Étape 4 : Entraînement du modèle Random Forest")
    df_rf = charger_rf()
    if df_rf is not None:
        X_train, X_test, y_train, y_test = prep_rf(df_rf)
        model_rf = entrainer_modele_rf(X_train, y_train)
        mae_rf, rmse_rf, r2_rf = eval_rf(model_rf, X_test, y_test)
        enregistrer_performances_csv("Random Forest", mae_rf, rmse_rf, r2_rf)

    print("📊 Étape 5 : Comparaison graphique des scores")
    pipeline_comparaison()

    print("📈 Étape 6 : Comparaison graphique des prédictions")
    comparer_predictions()

    print("✅ Pipeline complet exécuté avec succès.")

if __name__ == "__main__":
    pipeline_global()

from .processing_data import process_consumption_data
from .train_test import preparer_donnees, entrainer_modele, evaluer_modele, sauvegarder_modele

def download_data():
    # appele download_file.main()
    ...

def preprocessing(path_annuel, path_calendar):
    return process_consumption_data(path_annuel, path_calendar)

def split_features(df_filtre):
    X, y, df_model = preparer_donnees(df_filtre)
    return X, y, df_model

def train_model(X, y):
    model, X_train, X_val, y_train, y_val = entrainer_modele(X, y)
    return model, X_train, X_val, y_train, y_val

def evaluate(model, X_train, X_val, y_train, y_val):
    return evaluer_modele(model, X_train, X_val, y_train, y_val)

def save_model(model):
    return sauvegarder_modele(model)

from kedro.pipeline import Pipeline, node
from .nodes import download_data, preprocessing, split_features, train_model, evaluate, save_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(download_data, None, ["path_annuel", "path_calendar"]),
        node(preprocessing, ["path_annuel", "path_calendar"], "df_filtre"),
        node(split_features, "df_filtre", ["X", "y", "df_model"]),
        node(train_model, ["X", "y"], ["model", "X_train", "X_val", "y_train", "y_val"]),
        node(evaluate, ["model", "X_train", "X_val", "y_train", "y_val"], "evaluation_metrics"),
        node(save_model, "model", "model_path")
    ])

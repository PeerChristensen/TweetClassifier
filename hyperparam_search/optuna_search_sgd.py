from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from mlxtend.feature_selection import ColumnSelector
import pandas as pd
import optuna
# from optuna.visualization import plot_slice
# from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
import joblib
import os

import configparser
config = configparser.ConfigParser()
config.read("config.ini")


def load_data() -> pd.DataFrame:
    return pd.read_csv(config['DATA']['train_path'])


# Define sklearn compatible pipeline
pipeline = Pipeline([
    ('col_selector', ColumnSelector(cols=config['MODELLING']['input'], drop_axis=True)),
    ('vct', TfidfVectorizer()),
    ("clf", SGDClassifier())
])


def objective(trial):

    df = load_data()

    max_ngram = trial.suggest_int("max_ngram", 1, 3)
    parameters = {
        # vectorizer params
        "vct__use_idf": trial.suggest_categorical('vct__use_idf', ['True', 'False']),
        "vct__ngram_range": (1, max_ngram),
        "vct__stop_words": trial.suggest_categorical("vct__stop_words", ['english', None]),
        "vct__min_df": trial.suggest_float("vct__min_df", 0.0, 0.1),
        "vct__max_df": trial.suggest_float("vct__max_df", 0.9, 1.0),
        "vct__max_features": trial.suggest_int("vct__max_features", 50, 50000),

        # classifier params
        "clf__loss": trial.suggest_categorical("clf__loss", ["hinge", "log"]),
        "clf__penalty": trial.suggest_categorical("clf__penalty", ['l2', 'l1', 'elasticnet']),
        "clf__alpha": trial.suggest_float("clf__alpha", 0, 0.01)
    }

    pipeline_copy = clone(pipeline).set_params(**parameters)
    score = cross_val_score(
        estimator=pipeline_copy,
        X=df,
        y=df[config['MODELLING']['target']],
        n_jobs=-1,
        cv=5,
        scoring='roc_auc')
    return score.mean()


if __name__ == "__main__":

    if not os.path.isfile(config['SGD_ARTEFACTS']['study_path']):
        study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=10)
    print(f"\nAUC: {round(study.best_value,3)}\nParams: {study.best_params}")

    joblib.dump(study, config['SGD_ARTEFACTS']['study_path'])
    joblib.dump(study.best_params, config['SGD_ARTEFACTS']['best_params'])
    study.trials_dataframe().to_csv(config['SGD_ARTEFACTS']['trials_df_path'], index=False)

    plt1 = plot_optimization_history(study)
    plt1.write_html(config['SGD_ARTEFACTS']['plot_optimization_history_path'])

    plt2 = plot_parallel_coordinate(study)
    plt2.write_html(config['SGD_ARTEFACTS']['plot_parallel_coordinate_path'])

    plt3 = plot_param_importances(study)
    plt3.write_html(config['SGD_ARTEFACTS']['plot_param_importance_path'])

    plt4 = plot_param_importances(study,
                                  target=lambda t: t.duration.total_seconds(),
                                  target_name="duration")
    plt4.write_html(config['SGD_ARTEFACTS']['plot_param_duration_path'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
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
    ("clf", MultinomialNB())
])


def objective(trial):

    df = load_data()

    vct__ngram_range = trial.suggest_int("vct__ngram_range", 1, 3)
    parameters = {
        # vectorizer params
        "vct__use_idf": trial.suggest_categorical('vct__use_idf', ['True', 'False']),
        "vct__lowercase": trial.suggest_categorical('vct__lowercase', ['True', 'False']),
        "vct__ngram_range": (1, vct__ngram_range),
        "vct__stop_words": trial.suggest_categorical("vct__stop_words", ['english', None]),
        "vct__min_df": trial.suggest_float("vct__min_df", 0.0, 0.1),
        "vct__max_df": trial.suggest_float("vct__max_df", 0.9, 1.0),
        "vct__max_features": trial.suggest_int("vct__max_features", 50, 50000),

        # classifier params
        "clf__fit_prior": trial.suggest_categorical("clf__fit_prior", ["True", "False"]),
        "clf__alpha": trial.suggest_float("clf__alpha", 0, 100)
    }

    pipeline_copy = clone(pipeline).set_params(**parameters)
    score = cross_val_score(
        estimator=pipeline_copy,
        X=df,
        y=df[config['MODELLING']['target']],
        n_jobs=-1,
        cv=int(config['MODELLING']['cv_folds']),
        scoring='roc_auc')
    return score.mean()


if __name__ == "__main__":

    if not os.path.isfile(config['NB_ARTEFACTS']['study_path']):
        study = optuna.create_study(direction="maximize")
    else:
        study = joblib.load(config['NB_ARTEFACTS']['study_path'])

    study.optimize(objective, n_trials=int(config['MODELLING']['n_trials']))

    joblib.dump(study, config['NB_ARTEFACTS']['study_path'])
    joblib.dump(study.best_params, config['NB_ARTEFACTS']['best_params'])
    study.trials_dataframe().to_csv(config['NB_ARTEFACTS']['trials_df_path'], index=False)

    plt1 = plot_optimization_history(study)
    plt1.write_html(config['NB_ARTEFACTS']['plot_optimization_history_path'])

    plt2 = plot_parallel_coordinate(study)
    plt2.write_html(config['NB_ARTEFACTS']['plot_parallel_coordinate_path'])

    plt3 = plot_param_importances(study)
    plt3.write_html(config['NB_ARTEFACTS']['plot_param_importance_path'])

    plt4 = plot_param_importances(study,
                                  target=lambda t: t.duration.total_seconds(),
                                  target_name="duration")
    plt4.write_html(config['NB_ARTEFACTS']['plot_param_duration_path'])

    print("Optimization of NB classifier - DONE")
    print(f"\nAUC: {round(study.best_value,3)}\nParams: {study.best_params}")
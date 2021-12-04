
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from mlxtend.feature_selection import ColumnSelector

import pandas as pd
import matplotlib.pyplot as plt
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":

    train = pd.read_csv(config['DATA']['train_path'])

    params = joblib.load(config['SGD_ARTEFACTS']['best_params'])
    params['vct__ngram_range'] = (1, params['vct__ngram_range'])

    pipeline = Pipeline([
        ('col_selector', ColumnSelector(cols=config['MODELLING']['input'], drop_axis=True)),
        ('vct', TfidfVectorizer()),
        ("clf", SGDClassifier(loss="log"))
    ])

    pipeline.\
        set_params(**params).\
        fit(train, train[config['MODELLING']['target']])

    with open(config['MODELS']['sgd_model'], "wb") as f:
        joblib.dump(pipeline, f)

    print("Training of best SGD classifier - DONE")



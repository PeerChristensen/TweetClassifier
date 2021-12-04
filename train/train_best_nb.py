
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  classification_report

from mlxtend.feature_selection import ColumnSelector

import pandas as pd
import matplotlib.pyplot as plt
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":

    train = pd.read_csv(config['DATA']['test_path'])

    params = joblib.load(config['NB_ARTEFACTS']['best_params'])
    params['vct__ngram_range'] = (1, params['vct__ngram_range'])

    pipeline = Pipeline([
        ('col_selector', ColumnSelector(cols=config['MODELLING']['input'], drop_axis=True)),
        ('vct', TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    pipeline.\
        set_params(**params).\
        fit(train, train[config['MODELLING']['target']])

    with open(config['MODELS']['nb_model'], "wb") as f:
        joblib.dump(pipeline, f)

    print("Training of best NB classifier - DONE")




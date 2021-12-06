import pandas as pd
import seaborn as sns

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

# -------------------------------------------------------------------------------------------------
# SGD
# -------------------------------------------------------------------------------------------------

df = pd.read_csv(config['SGD_ARTEFACTS']['trials_df_path'])


sns.lmplot(x='number', y='params_clf__alpha', data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_clf__penalty", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__lowercase", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__ngram_range", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__stop_words", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__use_idf", data=df, ci=None, lowess=True, truncate=True)

# -------------------------------------------------------------------------------------------------
# NB
# -------------------------------------------------------------------------------------------------

df = pd.read_csv(config['NB_ARTEFACTS']['trials_df_path'])

sns.lmplot(x='number', y='params_clf__alpha', data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_clf__fit_prior", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='params_vct__min_df', data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__lowercase", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__ngram_range", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__stop_words", data=df, ci=None, lowess=True, truncate=True)

sns.lmplot(x='number', y='value', hue="params_vct__use_idf", data=df, ci=None, lowess=True, truncate=True)

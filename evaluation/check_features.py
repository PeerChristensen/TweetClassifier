import pandas as pd
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":

    # SGD
    pipeline = joblib.load(config['MODELS']['sgd_model'])

    model = pipeline.named_steps["clf"]

    feature_names = pipeline.named_steps["vct"].get_feature_names_out()
    feature_coefs = pipeline.named_steps["clf"].coef_

    coefficient_df = pd.DataFrame()
    coefficient_df["feature"] = feature_names
    coefficient_df["coef"] = feature_coefs[0]
    coefficient_df["abs_coef"] = coefficient_df["coef"].abs()

    coefficient_df.sort_values(by='abs_coef', ascending=False, inplace=True)
    coefficient_df.to_csv(config['SGD_EVAL']['model_features_path'], index=False)

    # NB
    pipeline = joblib.load(config['MODELS']['nb_model'])

    model = pipeline.named_steps["clf"]

    feature_names = pipeline.named_steps["vct"].get_feature_names_out()
    feature_coefs = pipeline.named_steps["clf"].coef_

    coefficient_df = pd.DataFrame()
    coefficient_df["feature"] = feature_names
    coefficient_df["coef"] = feature_coefs[0]
    coefficient_df["abs_coef"] = coefficient_df["coef"].abs()

    coefficient_df.sort_values(by='abs_coef', ascending=False, inplace=True)
    coefficient_df.to_csv(config['NB_EVAL']['model_features_path'], index=False)

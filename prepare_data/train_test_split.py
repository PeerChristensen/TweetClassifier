import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

df = pd.read_csv(config['DATA']['cleaned_data_path'])

train = df.sample(round(len(df)*.7), random_state=1)
test = df[~df.index.isin(train.index)]

train.to_csv(config['DATA']['train_path'], index=False)
test.to_csv(config['DATA']['test_path'], index=False)

print("train_test_split - DONE")
print(f"train and test data saved in data folder")

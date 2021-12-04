import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":
    df = pd.read_csv(config['DATA']['original_data_path'])

    # remove retweets
    df = df[df["isRetweet"] == "f"]

    # keep id, text and device columns
    df = df[['id', 'text', 'device']]

    # clean device column
    df["device"] = df["device"].str.replace("Twitter for ","")

    # keep only Android and iPhone
    devices = ["iPhone", "Android"]
    df = df[df["device"].isin(devices)]

    # Write csv
    df.to_csv(config['DATA']['cleaned_data_path'], index=False)

    print("clean_data - DONE")
    print(f"clean data saved as {config['DATA']['cleaned_data_path']}")


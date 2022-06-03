import requests

import zipfile
import pandas as pd

DATA_URL_ELEC = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
DATA_URL_BIKE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'

def download(url_elec, url_bike):
    r = requests.get(url_elec)
    with open("data/" + url_elec.split('/')[-1], "wb") as file:
        file.write(r.content)

    with zipfile.ZipFile('data/LD2011_2014.txt.zip', 'r') as archive:
        archive.extractall('./data')
        df = pd.read_csv("data/LD2011_2014.txt", delimiter=";", infer_datetime_format=True, parse_dates=[0],
                         index_col=0, decimal=",")
        df = df.resample("1h").mean()[:-1]
        df.index.name = "time"
        df.to_csv("data/elec.csv")

    r = requests.get(url_bike)
    with open("data/" + url_bike.split('/')[-1], "wb") as file:
        file.write(r.content)

    with zipfile.ZipFile('data/Bike-Sharing-Dataset.zip', 'r') as archive:
        archive.extractall('./data')
        df = pd.read_csv("data/hour.csv", delimiter=",", infer_datetime_format=True, parse_dates=["dteday"],
                         index_col="dteday")
        df.index = df.index + pd.Series(map(lambda v: pd.Timedelta(f"{v}h"), df['hr'].values))
        df = df.resample("1h").interpolate()
        df.index.name = "time"
        df.to_csv("data/bike.csv")

if __name__ == '__main__':
    download(DATA_URL_ELEC, DATA_URL_BIKE)

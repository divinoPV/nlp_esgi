import pandas as pd


def make_dataset(filename):
    return pd.read_csv(filename, encoding='utf8', delimiter=';')

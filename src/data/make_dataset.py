from pandas import read_csv


def make_dataset(filename):
    return read_csv(filename, encoding='utf8', delimiter=';')

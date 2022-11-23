import pandas as pd


def drop_duplicates(data_path, reset_index=True):
    '''
    Удаляет дубликаты и реиндексирует
    '''
    data = pd.read_csv(data_path, index_col=0)
    data.drop_duplicates(inplace=True)
    if reset_index:
        data.reset_index(drop=True, inplace=True)
    data.to_csv(data_path)

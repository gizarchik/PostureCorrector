import pandas as pd
import numpy as np

import src.config as CFG

def extract_metrics_full():
    def standart_metrics(data):
        extracted = data.groupby(by='_id', sort=False).agg(['mean', 'std', 'max', 'min'])
        extracted.columns = extracted.columns.map('_'.join)
        
        return extracted.astype(np.float64)
    
    
    def sma(data):
        data = data.copy()
        data['magnitude'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        SMA = data.groupby(by='_id', sort=False).agg(['mean'])['magnitude']
        
        return SMA.rename(columns={'mean':'sma'})
    
    
    def energy_metric(data):
        def energy(series):
            series = series**2
            return np.mean(series)
        extracted = data.groupby(by='_id', sort=False).agg([energy])
        extracted.columns = extracted.columns.map('_'.join)
        
        return extracted.astype(np.float64)
    
    
    def iqr_metric(data):
        def iqr(series):
            return np.quantile(series, 0.75) - np.quantile(series, 0.25)

        extracted = data.groupby(by='_id', sort=False).agg([iqr])
        extracted.columns = extracted.columns.map('_'.join)
        
        return extracted.astype(np.float64)

    
    def corr_metric(data):
        pairs = [['x', 'y'], ['x', 'z'], ['y', 'z']]
        corr_df = pd.DataFrame()

        for pair in pairs:
            corr_ser = data.groupby(by='_id', sort=False)[pair].corr().iloc[0::2,-1]
            corr_ser.index = corr_ser.index.droplevel(1)

            col_name = ''.join(pair) + '_corr'
            corr_df[col_name] = corr_ser

        return corr_df
    
    
    data_df = pd.read_csv(CFG.RAW_DATA_PATH, index_col=0)
    person_df = pd.read_csv(CFG.RAW_PERSON_PATH, index_col='_id')
    
    extracted = standart_metrics(data_df)
    extracted['sma'] = sma(data_df)['sma']
    extracted = pd.concat([extracted, energy_metric(data_df)], axis=1)
    extracted = pd.concat([extracted, iqr_metric(data_df)], axis=1)
    extracted = pd.concat([extracted, corr_metric(data_df)], axis=1)
    
    extracted.to_csv(CFG.METRICS_DATA_FULL_PATH)
    

def extract_metrics_windowed(window_size = CFG.WINDOW_SIZE, step = CFG.STEP):
    def standart_metrics(data):
        def energy(series):
            series = series**2
            return np.mean(series)

        def iqr(series):
            return np.quantile(series, 0.75) - np.quantile(series, 0.25)

        extracted = data_df.groupby(by='_id', sort=False)\
            .rolling(window=window_size, min_periods=window_size)\
            .agg(['mean', 'std', 'max', 'min', energy, iqr])\
            .dropna(axis=0, how='any')[::step]\
            .drop('_id', axis=1)
        extracted.columns = extracted.columns.map('_'.join)

        return extracted.astype(np.float64)
    
    
    def sma(data):
        data = data.copy()
        data['magnitude'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        SMA = data.groupby(by='_id', sort=False)\
            .rolling(window=window_size, min_periods=window_size)\
            .agg(['mean'])\
            .dropna(axis=0, how='any')[::step]\
            .drop('_id', axis=1)['magnitude']
        return SMA.rename(columns={'mean':'sma'})
    
    
    def corr_metric(data):
        pairs = [['x', 'y'], ['x', 'z'], ['y', 'z']]
        corr_df = pd.DataFrame()

        for pair in pairs:
            corr_ser = data.groupby(by='_id', sort=False)[pair]\
            .rolling(window=window_size)\
            .corr().iloc[0::2,-1]\
            .dropna(axis=0, how='any')[::step]

            corr_ser.index = corr_ser.index.droplevel(2)

            col_name = ''.join(pair) + '_corr'
            corr_df[col_name] = corr_ser

        return corr_df
    
    
    
    data_df = pd.read_csv(CFG.RAW_DATA_PATH, index_col=0)
    person_df = pd.read_csv(CFG.RAW_PERSON_PATH, index_col='_id')
    
    extracted = standart_metrics(data_df)
    extracted['sma'] = sma(data_df)['sma']
    extracted = pd.concat([extracted, corr_metric(data_df)], axis=1)
    
    extracted.to_csv(CFG.METRICS_DATA_WINDOWED_PATH)


def add_pos1_features():
    data = pd.read_csv(CFG.METRICS_DATA_FULL_PATH, index_col='_id')
    person = pd.read_csv(CFG.RAW_PERSON_PATH, index_col='_id')
    windowed_data = pd.read_csv(CFG.METRICS_DATA_WINDOWED_PATH, index_col=[0, 1])
    
    added_idx = person[(person['position'] == 1) & (person['is_valid'] == 1)].index
    rest_idx = person[~((person['position'] == 1) & (person['is_valid'] == 1))].index

    added_person = person.loc[added_idx].copy()
    rest_person = person.loc[rest_idx].copy()
    
    id_cols = ['height', 'mass', 'age', 'sex']
    added_data = pd.DataFrame()
    added_col_names = list(map(lambda x: x + '_pos1', data.columns))
    
    for added_idx, x in added_person[id_cols].iterrows():
        idx = person[np.all(person[id_cols] == x, axis=1)].index

        added_row = data.loc[added_idx]
        rows = data.loc[idx]
        rows[added_col_names] = added_row

        added_data = pd.concat([added_data, rows])

    added_data = added_data.iloc[::,22:]
    added_data = pd.concat([added_data, person], axis=1)
    
    new_data = pd.concat([windowed_data.reset_index(level=1), added_data], axis=1)\
                .dropna()\
                .set_index('level_1', append=True)
    
    new_data.to_csv(CFG.DATASET_PATH)
    
    
def extract_metrics():
    extract_metrics_full()
    extract_metrics_windowed()
    add_pos1_features()

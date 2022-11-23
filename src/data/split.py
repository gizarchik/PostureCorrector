import sys
sys.path.append('../')
sys.path.append('./')

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split as train_test_split_sklearn

from src.config import RAW_PERSON_PATH, TRAIN_ID_PATH, TEST_ID_PATH


def prepare_split(raw_person_data = RAW_PERSON_PATH):
    '''
    Разделяет выборку на обучающую и тестовую часть по людям. 
    Сохраняет разделенные id в отдельных файлах по путям TRAIN_ID_PATH, TEST_ID_PATH
    
    params:
        row_person_data (str) - путь до датасета людей, чтобы их разделить.
    
    return:
        TRAIN_ID_PATH (str), TEST_ID_PATH (str) - пути до файлов с разделенными id
    '''
    
    person_df = pd.read_csv(raw_person_data, index_col='_id')
    person_df = person_df.drop(1)
    
    new_preson_df = person_df.reset_index()
    new_preson_df = new_preson_df.set_index(['height', 'mass', 'age'])
    
    indexes = list(new_preson_df.index.unique())
    X_train, X_test = train_test_split_sklearn(indexes, test_size=0.33, random_state=42)
    
    X_train_id = new_preson_df.loc[X_train, '_id'].reset_index(drop=True)
    X_train_id.to_csv(TRAIN_ID_PATH)
    
    X_test_id = new_preson_df.loc[X_test, '_id'].reset_index(drop=True)
    X_test_id.to_csv(TEST_ID_PATH)
    
    return TRAIN_ID_PATH, TEST_ID_PATH



def train_test_split(X, y):
    train_id = pd.read_csv('../'+TRAIN_ID_PATH, index_col=[0])['_id'].values
    test_id = pd.read_csv('../'+TEST_ID_PATH, index_col=[0])['_id'].values
    
    X_train, y_train = X.loc[train_id], y.loc[train_id]
    X_test, y_test = X.loc[test_id], y.loc[test_id]
    
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, X_test, y_train, y_test

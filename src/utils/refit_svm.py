import sys
sys.path.append('../')
sys.path.append('./')

import pandas as pd

from src.data.uniting import unite_datasets
from src.data.drop_duplicates import drop_duplicates
from src.data.metrics import extract_metrics
from src.data.split import prepare_split
from src.models.train import svm_train

import src.config as CFG

import argparse


def re_fit(new_data_path : str, new_person_path : str) -> None:
    '''
    Полный пайплайн ре-обучения, включает в себя:
    1) Удаление дубликатов из новых данных
    2) Добавления новых данных к старым
    3) Извлечение нужных метрик для обучения
    4) Разделение на обучающую и тестовую выборки
    5) Обучение
    6) TODO: Подсчет скоров и проверка проблем датасета
    
    args:
        new_data_path(str) - Путь до новых данных для подсчета метрик
        new_person_path(str) - Путь до новых данных о людях
    '''
    drop_duplicates(new_data_path)
    unite_datasets(CFG.RAW_DATA_PATH, CFG.RAW_PERSON_PATH, new_data_path, new_person_path)
    extract_metrics()
    prepare_split()
    svm_train()
    # check_mistakes()


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="SVM re-trainer")
    parser.add_argument("new_data_path", type=str, help="Path to new data")
    parser.add_argument("new_person_path", type=str, help="Path to new person data")
    args = parser.parse_args()
    
    re_fit(args.new_data_path, args.new_person_path)

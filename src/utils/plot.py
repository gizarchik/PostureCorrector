import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_acc_and_time(history, sorted_metrics):
    '''
    Отрисовывает график времени предсказаний и точностей, полученных в ходе обучения модели 
    на подмножестве отсортированных по значимости метрик.
    
    hisory[dict(list)] - словарь из двух листов:
        history['accuracy'] - лист значений точности
        history['time_predict'] - лист значений времени предсказаний
    sorted_metrics[list(str)] - лист сортированных по значимости метрик
    '''
    
    plt.figure(figsize=(16, 10))
    plt.plot(
        np.arange(5, len(sorted_metrics)),
        history['accuracy']
    )
    plt.title("График зависимости Accuracy от числа метрик")
    plt.xlabel("Число метрик")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.plot(
        np.arange(5, len(sorted_metrics)),
        history['time_predict']
    )
    plt.title("График зависимости времени предсказания от числа метрик")
    plt.xlabel("Число метрик")
    plt.ylabel("Время предсказания, c")
    plt.show()

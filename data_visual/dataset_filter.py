import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

positisions = [1, 2, 3]
valids = [0, 1]

ALLOWED_MEAN_DIF = 500


def plot_comparision(data_1, data_2, cut=True):
    """
    Сравнивает два временных ряда

    Параметры:
    1) cut - Если True, то обрезает данные с большим размером до меньшего.
    """

    data_1.set_index('_id')
    data_2.set_index('_id')

    if cut:
        min_size = np.min((data_1.shape[0], data_2.shape[0]))
        data_1 = data_1[:min_size].copy()
        data_2 = data_2[:min_size].copy()

    plt.figure(figsize=(15, 40))

    for i, col in enumerate(['x', 'y', 'z']):
        plt.subplot(4, 1, i + 1)
        plt.scatter(range(len(data_1[col])), data_1[col], c='blue', alpha=0.25, label='data_1')
        plt.scatter(range(len(data_2[col])), data_2[col], c='red', alpha=0.25, label='data_2')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel(col)
        plt.title('Сравнение по {}'.format(col))

    # Сравним по длине векторов
    vec_1 = np.sqrt(data_1['x'] ** 2 + data_1['y'] ** 2 + data_1['z'] ** 2)
    vec_2 = np.sqrt(data_2['x'] ** 2 + data_2['y'] ** 2 + data_2['z'] ** 2)

    plt.subplot(4, 1, 4)
    plt.scatter(range(len(vec_1)), vec_1, c='blue', alpha=0.25, label='data_1')
    plt.scatter(range(len(vec_2)), vec_2, c='green', alpha=0.25, label='data_2')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title('Сравенение по r')

    plt.tight_layout()
    plt.show()


def is_mean_difference_allowed(corr_value: int, inc_value: int):
    return corr_value - inc_value > ALLOWED_MEAN_DIF


def check_separability(data_file: str, person_file: str) -> None:
    data_df = pd.read_csv(data_file)
    person_df = pd.read_csv(person_file, index_col='_id')
    person_df = person_df.drop(1)

    z_data = data_df[['_id', 'z']]
    grouped_z = z_data.groupby(by='_id', sort=False).mean()

    for pos in positisions:
        bad_pos_mask = person_df[(person_df['position'] == pos) & (person_df['is_valid'] == 0)].index
        good_pos_mask = person_df[(person_df['position'] == pos) & (person_df['is_valid'] == 1)].index

        incorrect_pos_data = list(grouped_z.loc[bad_pos_mask].iterrows())
        correct_pos_data = list(grouped_z.loc[good_pos_mask].iterrows())

        for (inc_sample, corr_sample) in zip(incorrect_pos_data[1:], correct_pos_data[1:]):
            if not is_mean_difference_allowed(corr_sample[1]['z'], inc_sample[1]['z']):
                corr_sample_index = corr_sample[0]
                inc_sample_index = inc_sample[0]
                print(f'BAD MEASURE: correct_id: {corr_sample_index}, incorrect_id: {inc_sample_index}')
                corr_data = data_df.loc[data_df['_id'] == corr_sample_index]
                inc_data = data_df.loc[data_df['_id'] == inc_sample_index]
                plot_comparision(corr_data, inc_data, cut=True)


check_separability('../data.csv', '../person.csv')
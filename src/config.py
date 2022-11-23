# Путь до датасета с метриками, посчитанным по окнам
DATASET_PATH = "datasets/added_pos1_metrics_windowed.csv"

TRAIN_ID_PATH = "datasets/X_train_id.csv"
TEST_ID_PATH = "datasets/X_test_id.csv"

RAW_DATA_PATH = "datasets/data.csv"
RAW_PERSON_PATH = "datasets/person.csv"

METRICS_DATA_FULL_PATH = 'datasets/metrics_data_full.csv'
METRICS_DATA_WINDOWED_PATH = 'datasets/windowed_metrics_data.csv'

ARCHIVE_PATH = 'datasets/archive/'

# for extract windowed metrics
WINDOW_SIZE = 100
STEP = 25

# svm config
max_iter = 1000
C = 8

# sorted metrics by CatBoost selection
sorted_metrics = ['z_mean',
 'y_mean',
 'z_energy',
 'x_mean_pos1',
 'y_energy',
 'sma',
 'z_min',
 'y_min_pos1',
 'x_min_pos1',
 'x_energy',
 'y_mean_pos1',
 'y_max_pos1',
 'x_mean',
 'mass',
 'y_iqr_pos1',
 'y_min',
 'xz_corr_pos1',
 'sma_pos1',
 'y_max',
 'z_max',
 'x_min',
 'y_iqr',
 'z_iqr_pos1',
 'x_iqr',
 'z_max_pos1',
 'x_iqr_pos1',
 'xy_corr_pos1',
 'x_max_pos1',
 'xz_corr',
 'z_std_pos1',
 'z_min_pos1',
 'x_energy_pos1',
 'x_std_pos1',
 'yz_corr_pos1',
 'y_std',
 'z_std',
 'x_std',
 'xy_corr',
 'x_max',
 'z_iqr',
 'z_energy_pos1',
 'z_mean_pos1',
 'y_std_pos1',
 'height',
 'yz_corr',
 'y_energy_pos1']


# path to model weights
SVM_WEIGHTS_PATH = 'Model weights/svm_weights.json'

def svm_train():
    '''
    '''
    import sys
    sys.path.append('../')
    sys.path.append('./')

    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler

    from sklearn.metrics import classification_report

    from tqdm import tqdm

    import json

    from src.utils.plot import plot_acc_and_time
    from src.config import DATASET_PATH

    from src.models.SVM import CustomSVM
    from src.data.split import train_test_split
    from src.config import sorted_metrics
    from src.config import max_iter, C
    from src.config import SVM_WEIGHTS_PATH
    
    
    data = pd.read_csv(DATASET_PATH, index_col=[0, 1])
    data = data.reset_index(level=1, drop=True)
    data = data.drop(['position', 'age'], axis=1)
    print(data.shape)
    
    y = data['is_valid']
    X = data.drop('is_valid', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X[sorted_metrics[:8]], y)
    X_train = X_train[::3]
    y_train = y_train[::3]
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.half)
    y_train = y_train.to_numpy(dtype=np.half)
    
    X_test = scaler.transform(X_test).astype(np.half)
    y_test = y_test.to_numpy(dtype=np.half)
    
    svc = CustomSVM(max_iter=max_iter)
    svc.fit(X_train, y_train[:, np.newaxis])
    
    y_pred = svc.predict(X_test)
    target_names = ['crooked', 'straight']
    print(classification_report(y_test.astype(np.int), y_pred.astype(np.int)), target_names)
    
    svm_params = {'X_train':svc._X_train.tolist(), 'weights':svc.params_.tolist()}
    scaler_params = {'mean':scaler.mean_.tolist(), 'std':scaler.scale_.tolist()}
    params = {'svm':svm_params, 'scaler':scaler_params}
    
    with open(SVM_WEIGHTS_PATH, 'w+') as fp:
        json.dump(params, fp)

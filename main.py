import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
from pandas.errors import SettingWithCopyWarning

from utils.datasets import get_data_openml

import argparse

from utils.helpers import get_model_and_params_grid, hyperparams_tuning

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str,
    default='climate',
    choices=[
        'climate',
        'metal',
        'breast-c',
        'breast-t',
        'ilpd',
        'heart-l',
        'marketing',
        'heart-h',
        'chat',
        'seismic',
        'thoracic',
        'profb',
        'australian',
        'glass',
        'dmft',
        'credit',
        'kc2',
        'cmc',
        'primary-tumor',
        'diabetes',
        'sa-heart',
        'ecoli',
        'spect',
        'apnea',
        'sensory',
        'backache',
    ],
    help='name of the dataset')
parser.add_argument('--test_size', type=float,
                    default=0.3, help='test set proportion rangin from 0.1 to 1.0')
parser.add_argument(
    '--model', type=str, default='AdaBoostSDM',
    choices=['AdaBoost', 'AdaBoostSDM', 'LapRLS', 'LapSVM', 'TabNet'],
    help='classifier')
parser.add_argument('--SS', type=int,
                    default=20, help='number of random shuffle and splits')

opt = parser.parse_args()


def main():

    np.random.seed(42)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print('hello')

    dataset_name = opt.dataset
    model_name = opt.model
    test_size = opt.test_size
    n_splits = opt.SS

    data, labels, cat_dims, cat_idxs = get_data_openml(dataset_name)

    if model_name in ['AdaBoost', 'AdaBoostSDM', 'LapSVM']:
        labels = labels * 2 - 1

    final_accs = []

    model, param_grid = get_model_and_params_grid(model_name, cat_idxs, cat_dims)

    count_splits = 0
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        count_splits += 1

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_name in ['AdaBoostSDM', 'LapRLS', 'LapSVM']:

            search = hyperparams_tuning(model, param_grid, X_train_scaled, y_train)
            test_acc = search.score(X_test_scaled, y_test)

        if model_name == 'AdaBoost':

            model.fit(X_train_scaled, y_train)
            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)

        if model_name == 'TabNet':
            model.fit(X_train, y_train, batch_size=16)
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)

        final_accs.append(test_acc)

        print(f'splits completed: {count_splits} / {n_splits}')

    print('Result: mean acc: {:.1f}, and std: {:.1f}'.format(
        np.mean(final_accs)*100, np.std(final_accs)*100))


if __name__ == '__main__':
    main()

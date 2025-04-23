import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml


def get_data_openml(name):
    ''' get dataset from openml
    Args:
        name: the name of dataset

    Return:
        data: the set of samples
        labels: the labels of samples in data

    '''
    datasets = {
        'climate': 1467,
        'metal': 1447,
        'breast-c': 23499,
        'liver': 8,
        'ilpd': 1480,
        'heart-l': 1512,
        'marketing': 986,
        'heart-h': 1565,
        'chat': 820,
        'seismic': 1500,
        'thoracic': 1506,
        'profb': 470,
        'australian': 40981,
        'glass': 41,
        'dmft': 469,
        'credit': 31,
        'kc2': 1063,
        'cmc': 23,
        'primary-tumor': 1003,
        'diabetes': 37,
        'sa-heart': 1498,
        'ecoli': 39,
        'spect': 336,
        'apnea': 765,
        'sensory': 826,
        'backache': 463,
    }

    dataset = fetch_openml(as_frame=True, data_id=datasets[name], parser="pandas")

    labels_train = dataset.target
    df = dataset.data

    cat_dims = []
    cat_idxs = []
    idx = 0
    for col_name in df.columns:
        if str(df[col_name].dtype) == 'category' or str(df[col_name].dtype) == 'object':
            df[col_name] = df[col_name].astype('category')

            # tabnet
            l_enc = LabelEncoder()

            df[col_name] = l_enc.fit_transform(df[col_name].values)
            cat_dims.append(len(l_enc.classes_))

            cat_idxs.append(idx)
        idx += 1

    nan_indexes = np.array(df[df.isna().any(axis=1)].index)

    has_missing = False
    if len(nan_indexes) != 0:
        has_missing = True

    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels_train)

    data = df.to_numpy()

    data = data[np.isin(labels, [0, 1])]
    labels = labels[np.isin(labels, [0, 1])]

    print('\nDataset: ', name)
    print('Has missing: ', has_missing)
    print('The number of classes: ', len(np.unique(labels, return_counts=True)[0]))
    samples_per_class = [f'class {i}: {samples}' for i,
                         samples in enumerate(np.unique(labels, return_counts=True)[1])]
    print('The number of samples per class: \n', ', '.join(samples_per_class), '\n')

    return data, labels, cat_dims, cat_idxs

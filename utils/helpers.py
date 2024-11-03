import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from models.adaboost import AdaBoost

from models.adaboost_dsm import AdaBoostSDM
from models.laprls import LapRLS
from models.lapsvm import LapSVM

import torch
from pytorch_tabnet.tab_model import TabNetClassifier


def get_model_and_params_grid(model_name, cat_idxs, cat_dims):
    base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)

    M = 20
    if model_name == 'AdaBoostSDM':
        model = AdaBoostSDM(base_model=base_learner, max_models=M)

        param_grid = {
            "clf__sigma_percentile": [10, 20, 30, 40, 50, 60, 70],
            "clf__lambda_G": [5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, ],
        }
    elif model_name == 'LapRLS':
        model = LapRLS()
        param_grid = {
            "clf__lambda_k": [0.001, 0.01, 0.1, 1],
            "clf__lambda_u": [0.001, 0.01, 0.1, 1],
            "clf__n_neighbors": [5, 7, 9, 11],
        }
    elif model_name == 'LapSVM':
        model = LapSVM()
        param_grid = {
            "clf__lambda_k": [0.001, 0.01, 0.1, 1],
            "clf__lambda_u": [0.001, 0.01, 0.1, 1],
            "clf__n_neighbors": [5, 7, 9, 11],
        }
    elif model_name == 'AdaBoost':
        model = AdaBoost(base_model=base_learner, max_models=M)
        param_grid = None

    elif model_name == 'TabNet':
        model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims,)
        param_grid = None

    return model, param_grid


def hyperparams_tuning(model, param_grid, X_train, y_train):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline(steps=[("clf", model)])

    search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=5)
    search.fit(X_train, y_train)

    return search.best_estimator_

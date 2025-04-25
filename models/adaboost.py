import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import copy


class AdaBoost(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 base_model=DecisionTreeClassifier(max_depth=1, random_state=42),
                 max_models=15):

        self.base_model = base_model
        self.max_models = max_models

    def fit(self, X, y, ):
        ''' Fit model'''

        # Initialise
        self.models = []
        self.weights = []

        for t in range(self.max_models):
            if t == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = w_i * np.exp(a * (np.not_equal(y, k)).astype(int))
                w_i = w_i * np.exp(-a * (np.equal(y, k)).astype(int))
                sample_weights = w_i/np.sum(w_i)

            # normalize weights
            sample_weights = w_i/np.sum(w_i)

            # Fit model to dataset
            clf = copy.deepcopy(self.base_model)

            clf.fit(X, y, sample_weights)
            # Make predictions
            k = clf.predict(X)

            err = sum(w_i * (np.not_equal(y, k)).astype(int))/sum(w_i)
            a = 0.5*np.log((1-err)/err)

            if a < 0:
                print('Problematic convergence of the model. a<0')
                break

            # Save model
            self.models.append(clf)
            # save weights
            self.weights.append(a)
            # Update

            if err <= 0.01:
                break

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        # Predict weighting each model
        for i in range(len(self.models)):
            preds = np.add(preds, self.weights[i]*self.models[i].predict(X))
        preds = np.array(list(map(lambda x: 1 if x > 0 else -1, preds)))
        preds = preds.astype(int)
        return preds

    def scores(self, X):
        preds = np.zeros(X.shape[0])
        # Predict weighting each model
        for i in range(len(self.models)):
            preds = np.add(preds, self.weights[i]*self.models[i].predict(X))

        return preds

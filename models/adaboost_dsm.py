import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import copy
from scipy import sparse
from sklearn.metrics import pairwise_distances


class AdaBoostSDM(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 base_model=DecisionTreeClassifier(max_depth=1, random_state=42),
                 lambda_G=0.001,
                 sigma_percentile=20,
                 max_models=15):

        self.base_model = base_model
        self.lambda_G = lambda_G
        self.sigma_percentile = sigma_percentile
        self.max_models = max_models

    def fit(self, X, y, ):
        ''' Fit model'''

        # comptue weighted adjacency matrix W

        dists = pairwise_distances(X, metric='l2') * -1 / 2
        sigma = np.percentile(dists, self.sigma_percentile)

        self.W = np.exp(dists / sigma**2)
        self.W = sparse.csr_matrix(self.W)

        S_dis = np.not_equal(y[:, None], y[None, :])
        S_sim = np.equal(y[:, None], y[None, :])

        masked_W_sim = np.where(S_sim, self.W.todense(), 0)
        masked_W_dis = np.where(S_dis, self.W.todense(), 0)

        # Initialise
        self.models = []
        self.weights = []
        C = np.zeros(X.shape[0])

        for t in range(self.max_models):

            Cij_plus = C[:, np.newaxis] + C
            Cij_minus = C[:, np.newaxis] - C
            Cji_minus = C - C[:, np.newaxis]

            # comptue p_i and q_i for each data point using Eq. 15 and Eq. 16 from the paper, respectively
            p_1 = np.exp(-2*C)*(y == 1)

            p_2 = np.einsum('ij, ij -> i', masked_W_sim, np.exp(Cji_minus)) * self.lambda_G / 2
            p_3 = np.einsum('ij, ij -> i', masked_W_dis, np.exp(-Cij_plus)) * self.lambda_G / 2

            p2 = np.add(p_2, p_3)
            p = np.add(p_1, p2)

            q_1 = np.exp(2*C)*(y == -1)

            q_2 = np.einsum('ij, ij -> i', masked_W_sim, np.exp(Cij_minus)) * self.lambda_G / 2
            q_3 = np.einsum('ij, ij -> i', masked_W_dis, np.exp(Cij_plus)) * self.lambda_G / 2

            q2 = np.add(q_2, q_3)
            q = np.add(q_1, q2)

            pi = p*(y == 1)
            qi = q*(y == -1)

            # compute weights (Eq. 17 in the paper)
            w_i = pi+qi

            # normalize weights
            sample_weights = w_i/np.sum(w_i)

            # Fit model to dataset
            clf = copy.deepcopy(self.base_model)

            clf.fit(X, y, sample_weights)
            # Make predictions
            k = clf.predict(X)

            # Compute alpha (Eq. 24 in the paper)
            corr = (np.dot(p, k == 1) + np.dot(q, k == -1))
            err = (np.dot(p, k == -1) + np.dot(q, k == 1))
            a = 0.25*np.log(corr/err)

            if a < 0:
                break

            # Save model
            self.models.append(clf)
            # save weights
            self.weights.append(a)
            # Update

            if err <= 0.01:
                break

            C = np.add(C, a*k)

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        # Predict weighting each model
        for i in range(len(self.models)):
            preds = np.add(preds, self.weights[i]*self.models[i].predict(X))
        preds = np.array(list(map(lambda x: 1 if x > 0 else -1, preds)))
        preds = preds.astype(int)
        return preds

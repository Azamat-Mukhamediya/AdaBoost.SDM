
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

from scipy.optimize import minimize
from scipy import sparse


class LapSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, kernel=rbf_kernel, lambda_k=0.1, lambda_u=0.1):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.neighbor_mode = 'connectivity'

    def fit(self, X, Y):
        # construct graph
        self.X = X
        Y = np.diag(Y)
        if self.neighbor_mode == 'connectivity':
            W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity', include_self=False)
            W = (((W + W.T) > 0) * 1)
        elif self.neighbor_mode == 'distance':
            W = kneighbors_graph(self.X, self.n_neighbors, mode='distance', include_self=False)
            W = W.maximum(W.T)
            W = sparse.csr_matrix(
                (np.exp(-W.data ** 2 / 4 / self.opt['t']),
                 W.indices, W.indptr),
                shape=(self.X.shape[0],
                       self.X.shape[0]))
        else:
            raise Exception()

        # Computing Graph Laplacian
        L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W

        # Computing K with k(i,j) = kernel(i, j)
        K = self.kernel(self.X)

        l = X.shape[0]
        u = 0
        J = np.identity(l)

        # Computing "almost" alpha
        almost_alpha = np.linalg.inv(2 * self.lambda_k * np.identity(l + u) +
                                     ((2 * self.lambda_u) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y)

        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        Q = (Q+Q.T)/2

        del W, L, K, J

        e = np.ones(l)
        q = -e

        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))

        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]

        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        # ===== Solving =====
        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad,
                            constraints=cons, bounds=bounds, method='SLSQP')['x']

        # Computing final alpha
        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, Q

        # Finding optimal decision boundary b using labeled data
        new_K = self.kernel(self.X, X)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        self.sv_ind = np.nonzero((beta_hat > 1e-7)*(beta_hat < (1/l-1e-7)))[0]

        ind = self.sv_ind[0]
        self.b = np.diag(Y)[ind]-f[ind]

    def decision_function(self, X):
        new_K = self.kernel(self.X, X)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f+self.b

    def predict(self, Xtest):

        Y_ = self.decision_function(Xtest)
        predictions = np.ones(Xtest.shape[0])
        predictions[Y_ < 0] = -1

        return predictions

    def scores(self, Xtest):

        Y_ = self.decision_function(Xtest)

        return Y_
    
    def predict_proba(self, Xtest):

        Y_ = self.decision_function(Xtest)

        return Y_

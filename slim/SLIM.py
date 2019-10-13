import numpy as np
from scipy.sparse import *

class SLIMRec(object):
    def __init__(self, train_matrix, lr=0.05, l2_reg=0.001, l1_reg=0.01, regression_iter=10):
        self.n_users, self.n_items = train_matrix.shape
        self.A = train_matrix
        self.ATA = train_matrix.T.dot(train_matrix)
        self.beta = l2_reg
        self.gamma = l1_reg
        self.lr = lr
        self.regression_iter = regression_iter

    def train(self):
        ret = []
        for item_idx in range(self.n_items):
            w = self.train_one(item_idx)
            ret.append(w)
        self.W = np.hstack(ret)

    def train_one(self, item_idx):
        w = np.random.random(size=(self.n_items, 1))
        w[item_idx, 0] = 0.0
        for i in range(self.regression_iter):
            grad = (self.ATA * w - self.ATA[:, item_idx])+ self.beta * w + self.gamma * (w / (1e-6 + np.abs(w)))
            w -= self.lr * grad
            w[w < 0] = 1e-6
            w[item_idx, 0] = 0.0
        w[w < 1e-6] = 0.0
        print(item_idx, self.A[:, item_idx].nnz, min(w), max(w))
        return w
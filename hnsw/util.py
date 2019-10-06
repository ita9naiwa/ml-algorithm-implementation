import numpy as np

def l2_dist(v1, v2):
    return (((v1 - v2) **2).sum())


def node_dist(n1, n2, method='l2'):
    if method == 'l2':
        return l2_dist(n1.vec, n2.vec)

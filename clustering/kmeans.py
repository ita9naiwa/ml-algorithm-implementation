import numpy as np

class KMeans(object):
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        
    def init(self, vectors):
        self.vectors = vectors
        self.num_vectors, self.dim = vectors.shape
        self.membership = np.random.randint(0, self.num_cluster, size=self.num_vectors)
        self.centroids = np.zeros(shape=(self.num_cluster, self.dim))
        self.update_centroid()

    def update_centroid(self):
        for k in range(self.num_cluster):
            self.centroids[k] = np.mean(self.get_vecs_clsuter_k(k), axis=0)

    def calc_loss(self):
        ret = 0
        for k in range(self.num_cluster):
            ret += np.sum((self.get_vecs_clsuter_k(k) - self.centroids[k]) ** 2)
        return ret

    def get_vecs_clsuter_k(self, k):
        return self.vectors[self.membership == k]

    def iter_once(self):
        for n in range(self.num_vectors):
            ret = np.sum((self.centroids - self.vectors[n]) ** 2, axis=1)
            self.membership[n] = np.argmin(ret)
        self.update_centroid()

    def iterate(self, num_iter):
        for iter in range(num_iter):
            self.iter_once()
            print("loss: %0.2f" % self.calc_loss())


def main():
    import pickle
    with open("../data/item_factors.pkl", "rb") as f:
        item_factors = pickle.load(f)
    model = KMeans(10)
    model.init(item_factors)
    model.iterate(30)
if __name__ == "__main__":
    main()

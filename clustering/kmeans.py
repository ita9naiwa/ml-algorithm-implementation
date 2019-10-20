import numpy as np
import matplotlib.pyplot as plt


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

    def plot(self, ax):
        colormap = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for k in range(self.num_cluster):
            ret =self.get_vecs_clsuter_k(k)
            x = ret[:, 0]
            y = ret[:, 1]
            ax.scatter(x, y, color=colormap[k], s=4)


    def iter_once(self):
        for n in range(self.num_vectors):
            ret = np.sum((self.centroids - self.vectors[n]) ** 2, axis=1)
            self.membership[n] = np.argmin(ret)
        self.update_centroid()

    def iterate(self, num_iter):
        fig, axes = plt.subplots(1, num_iter)
        for iter in range(num_iter):
            self.iter_once()
            print("loss: %0.2f" % self.calc_loss())
            self.plot(axes[iter])
        plt.show()

def get_random_samples(size=5000):
    half_size = size // 2
    r = np.random.random()
    x_0 = np.random.normal(-3, 1, size=half_size)
    x_1 = np.random.normal(3, 1, size=half_size)
    y_0 = np.random.normal(-3, 1, size=half_size)
    y_1 = np.random.normal(3, 1, size=half_size)
    x = np.hstack([x_0, x_1])
    np.random.shuffle(x)
    y = np.hstack([y_0, y_1])
    np.random.shuffle(y)
    return np.vstack([x, y]).T

def main():
    """
    import pickle
    with open("../data/item_factors.pkl", "rb") as f:
        item_factors = pickle.load(f)
    """
    data = get_random_samples(10000)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    print(data.shape)
    model = KMeans(4)
    model.init(data)
    model.iterate(5)
if __name__ == "__main__":
    main()

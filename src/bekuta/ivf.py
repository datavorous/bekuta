from .base import BaseIndex
import random as r

class IVFIndex(BaseIndex):
    def __init__(self, dim, metric="cosine", n_lists=100):
        super().__init__(dim, metric)
        self.n_lists = n_lists
        self.centroids = []
        self.inverted_lists = []
        self.inverted_ids = []
        self.is_trained = False

    def train(self, training_vectors):

        # we are picking random samples as centroids
        self.centroids = r.sample(training_vectors, self.n_lists)

        # nudges, im going with 25 iterations, which is a common choice for k-means
        for iters in range(25):

            # assigning vectors to the nearest centroid
            assignments = [[] for _ in range(self.n_lists)]

            for vec in training_vectors:
                nearest_index = self._find_nearest_centroid(vec)
                assignments[nearest_index].append(vec)

            # move centroids to the mean of their assigned vectors
            for i in range(self.n_lists):
                if assignments[i]:
                    self.centroids[i] = self._compute_centroid(assignments[i])

        self.is_trained = True
        
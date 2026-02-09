from .base import BaseIndex
from .kernels import SimilarityMetric

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

    def _find_nearest_centroid(self, vector):
        best_score = float("-inf")
        best_index = 0

        q = vector

        if self.metric == "cosine":
            q = SimilarityMetric.normalize(vector)

        for i, centroid in enumerate(self.centroids):
            score = SimilarityMetric.score(self.metric, q, centroid)
            if score > best_score:
                best_score = score
                best_index = i

        return best_index

    def _compute_mean(self, vectors):
        dim = len(vectors[0])
        mean = [0.0] * dim

        for vec in vectors:
            for i in range(dim):
                mean[i] += vec[i]

        for i in range(dim):
            mean[i] /= len(vectors)

        return mean

    def add(self, id, vector):
        if not self.is_trained:
            raise ValueError("CAN NOT ADD VECTORS BEFORE BEING TRAINED")

        v = vector
        if self.metric == "cosine":
            v = SimilarityMetric.normalize(vector)

        cluster_id = self._find_nearest_centroid(v)
        self.inverted_lists[cluster_id].append(v)
        self.inverted_ids[cluster_id].append(id)

    def __len__(self):
        total = 0
        for inverted_list in self.inverted_lists:
            total += len(inverted_list)
        return total

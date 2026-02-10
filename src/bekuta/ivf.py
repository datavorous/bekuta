from .base import BaseIndex
from .kernels import SimilarityMetric

import random as r
import heapq

class IVFIndex(BaseIndex):
    def __init__(self, dim, metric="cosine", n_lists=100, n_probe=1):
        super().__init__(dim, metric)
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.centroids = []
        self.inverted_lists = []
        self.inverted_ids = []
        self.is_trained = False

    def train(self, training_vectors):

        # we are picking random samples as centroids
        if len(training_vectors) < self.n_lists:
            raise ValueError("training_vectors must be >= n_lists")
        self.centroids = r.sample(training_vectors, self.n_lists)
        self.inverted_lists = [[] for _ in range(self.n_lists)]
        self.inverted_ids = [[] for _ in range(self.n_lists)]

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
                    self.centroids[i] = self._compute_mean(assignments[i])

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


    def search(self, query_vector, k):
        if not self.is_trained:
            raise ValueError("MUST BE TRAINED BEFORE SEARCHING")

        q = query_vector
        if self.metric == "cosine":
            q = SimilarityMetric.normalize(query_vector)

        # cluster_id = self._find_nearest_centroid(q)

        cluster_scores = []
        for i, centroid in enumerate(self.centroids):
            score = SimilarityMetric.score(self.metric, q, centroid)
            cluster_scores.append((score, i))

        cluster_scores.sort(reverse=True)
        clusters_to_search = [index for i, index in cluster_scores[:self.n_probe]]

        heap = []
        for cluster_id in clusters_to_search:
            # we just added a new loop around it
            for pos in range(len(self.inverted_lists[cluster_id])):
                vec = self.inverted_lists[cluster_id][pos]
                id_val = self.inverted_ids[cluster_id][pos]
                score = SimilarityMetric.score(self.metric, q, vec)

                if len(heap) < k:
                    heapq.heappush(heap, (score, id_val))
                else:
                    heapq.heappushpop(heap, (score, id_val))
        
        return [(id_val, s) for s, id_val in sorted(heap, reverse=True)]
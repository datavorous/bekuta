import heapq

from .utils import normalize
from .kernels import score
from .base import BaseIndex


class FlatIndex(BaseIndex):
    def __init__(self, dim, metric="cosine"):
        super().__init__(dim, metric)
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}

        def add(self, id, vector):
            v = vector
            if self.metric == "cosine":
                v = normalize(vector)

            pos = len(self.vectors)
            self.vectors.append(v)
            self.ids.append(id)
            self.ids_to_index[id] = pos

        def search(self, query_vector, k):
            q = query_vector
            if self.metric == "cosine":
                q = normalize(query_vector)

            heap = []
            for pos in range(len(self.vectors)):
                id = self.ids[pos]
                s = score(self.metric, q, self.vectors[pos])
                if len(heap) < k:
                    heapq.heappush(heap, (s, id))
                else:
                    heapq.heappushpop(heap, (s, id))

            return [(id, s) for s, id in sorted(heap, reverse=True)]

        def delete(self, id):
            if id not in self.ids_to_index:
                return

            pos = self.ids_to_index[id]
            last_pos = len(self.vectors) - 1

            if pos != last_pos:
                self.vectors[pos] = self.vectors[last_pos]
                self.ids[pos] = self.ids[last_pos]
                self.ids_to_index[self.ids[pos]] = pos

            self.vectors.pop()
            self.ids.pop()
            del self.ids_to_index[id]
            return True

        def __len__(self):
            return len(self.vectors)

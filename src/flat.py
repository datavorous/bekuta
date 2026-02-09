from .utils import normalize
from .kernels import score


class FlatIndex:
    def __init__(self, dim, metric="cosine"):
        self.dim = dim
        self.metric = metric
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}


def add(index, id, vector):
    v = vector
    if index.metric == "cosine":
        v = normalize(vector)

    pos = len(index.vectors)
    index.vectors.append(v)
    index.ids.append(id)
    index.ids_to_index[id] = pos

def add_batch(index, ids, vectors):
    for id, vector in zip(ids, vectors):
        add(index, id, vector)


def search(index, query_vector, k):
    q = query_vector
    if index.metric == "cosine":
        q = normalize(query_vector)

    results = []
    for pos in range(len(index.vectors)):
        id = index.ids[pos]
        s = score(index.metric, q, index.vectors[pos])
        results.append((id, s))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

def heap_search(index, query_vector, k):
    import heapq

    q = query_vector
    if index.metric == "cosine":
        q = normalize(query_vector)

    heap = []
    for pos in range(len(index.vectors)):
        id = index.ids[pos]
        s = score(index.metric, q, index.vectors[pos])
        if len(heap) < k:
            heapq.heappush(heap, (s, id))
        else:
            heapq.heappushpop(heap, (s, id))

    return [(id, s) for s, id in sorted(heap, reverse=True)]
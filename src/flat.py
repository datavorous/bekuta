from .utils import normalize

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

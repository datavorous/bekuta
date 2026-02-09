class BaseIndex:
    def __init__(self, dim, metric="cosine"):
        self.dim = dim
        self.metric = metric

    def add(self, id, vector):
        raise NotImplementedError

    def add_batch(self, ids, vectors):
        for i in range(len(ids)):
            self.add(ids[i], vectors[i])

    def search(self, query_vector, k):
        raise NotImplementedError

    def delete(self, id):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

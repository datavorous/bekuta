

class FlatIndex:
    def __init__(self, dim, metric="cosine"):
        self.dim = dim
        self.metric = metric 
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}
from .flat import FlatIndex
from .ivf import IVFIndex

class Engine:
    @staticmethod
    def create_index(dim, metric="cosine", index_type="flat", **kwargs):
        if index_type == "flat":
            return FlatIndex(dim, metric)
        elif index_type == "ivf":
            n_lists = kwargs.get('n_lists', 100)
            n_probe = kwargs.get('n_probe', 1)
            return IVFIndex(dim, metric, n_lists, n_probe)
        raise ValueError(f"Unknown index type: {index_type}")
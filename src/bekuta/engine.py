from .flat import FlatIndex


class Engine:
    @staticmethod
    def create_index(dim, metric="cosine", index_type="flat"):
        if index_type == "flat":
            return FlatIndex(dim, metric)
        raise ValueError(f"Unknown index type: {index_type}")

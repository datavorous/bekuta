from .flat import FlatIndex


class Engine:
    def create_index(dim, metric="cosine", index_type="flat"):
        if index_type == "flat":
            return FlatIndex(dim, metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

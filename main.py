from bekuta import Engine
from profiling import profile
import random


def make_dataset(count, dim):
    rng = random.Random(42)
    ids = [f"vec{i}" for i in range(count)]
    vectors = [[rng.random() for _ in range(dim)] for _ in range(count)]
    return ids, vectors


@profile
def build_flat(ids, vectors):
    index = Engine.create_index(dim=8, metric="l2", index_type="flat")
    index.add_batch(ids, vectors)
    return index


@profile
def build_ivf(ids, vectors):
    index = Engine.create_index(dim=8, metric="l2", index_type="ivf", n_lists=50, n_probe=4)
    index.train(vectors)
    for id_val, vec in zip(ids, vectors):
        index.add(id_val, vec)
    return index


@profile
def search_index(index, query):
    return index.search(query, k=5)


ids, vectors = make_dataset(count=5000, dim=8)

flat = build_flat(ids, vectors)
ivf = build_ivf(ids, vectors)

query = vectors[0]
flat_results = search_index(flat, query)
ivf_results = search_index(ivf, query)

print("\nFlat results:", flat_results[:3])
print("IVF results:", ivf_results[:3])
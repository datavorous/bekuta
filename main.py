from bekuta import Engine
from profiling import profile


@profile
def create_and_populate_index():
    engine = Engine.create_index(dim=3, metric="l2", index_type="flat")
    engine.add_batch(
        ["a", "b", "c"],
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.1, 1.1, 1.1]],
    )
    return engine


@profile
def run_search(engine):
    query = [0.8, 0.6, 0]
    results = engine.search(query, k=2)
    return results


engine = create_and_populate_index()

query = [0.8, 0.6, 0]

results = run_search(engine)

for id_val, score_val in results:
    print(f"id: {id_val}, score: {score_val}")

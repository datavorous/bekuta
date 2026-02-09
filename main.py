from src.engine import Engine

engine = Engine.create_index(dim=3, metric="l2", index_type="flat")

engine.add_batch(
    ["a", "b", "c"],
    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.1, 1.1, 1.1]],
)

query = [0.8, 0.6, 0]

results = engine.search(query, k=2)

for id_val, score_val in results:
    print(f"id: {id_val}, score: {score_val}")


"""
start = time.perf_counter()
results = engine.search(query, k=2)
end = time.perf_counter()

print(f"Search took {(end - start) * 1000:.3f}ms")

for id_val, score_val in results:
    print(f"id: {id_val}, score: {score_val}")

print("Before deletion:")
for id_val, vector in zip(engine.ids, engine.vectors):
    print(f"id: {id_val}, vector: {vector}")

engine.delete("b")

print("After deletion:")
for id_val, vector in zip(engine.ids, engine.vectors):
    print(f"id: {id_val}, vector: {vector}")
"""




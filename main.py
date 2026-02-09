from src.flat import FlatIndex, add, search, heap_search, add_batch
import time

index = FlatIndex(dim=3, metric="cosine")

batch_size = 100_000
for i in range(0, 10_000_000, batch_size):
    ids = [f"vec{j}" for j in range(i, min(i + batch_size, 10_000_000))]
    vectors = [[j % 10, (j // 10) % 10, (j // 100) % 10] for j in range(i, min(i + batch_size, 10_000_000))]

add_batch(index, ids, vectors)

query = [0.8, 0.6, 0]

start = time.perf_counter()
results = search(index, query, k=2)
end = time.perf_counter()

print(f"Search took {(end - start) * 1000:.3f}ms")

start = time.perf_counter()
results_heap = heap_search(index, query, k=2)
end = time.perf_counter()

print(f"Heap search took {(end - start) * 1000:.3f}ms")

for id_val, score_val in results:
    print(f"id: {id_val}, score: {score_val}")

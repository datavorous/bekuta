from src.flat import FlatIndex, add, search, heap_search
import time

index = FlatIndex(dim=3, metric="cosine")

for i in range(10_000_000):
    add(index, f"vec{i}", [i % 10, (i // 10) % 10, (i // 100) % 10])

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

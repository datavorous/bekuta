# LLM Generated Code : True

from bekuta import Engine
from profiling import memory_usage_mb
import random
import time

def make_dataset(count, dim, seed=42):
    rng = random.Random(seed)
    ids = [f"vec{i}" for i in range(count)]
    vectors = [[rng.random() for _ in range(dim)] for _ in range(count)]
    return ids, vectors

def build_flat(ids, vectors, dim):
    index = Engine.create_index(dim=dim, metric="l2", index_type="flat")
    index.add_batch(ids, vectors)
    return index

def build_ivf(ids, vectors, dim, n_lists):
    index = Engine.create_index(
        dim=dim, 
        metric="l2", 
        index_type="ivf", 
        n_lists=n_lists, 
        n_probe=1  # Will change this later
    )
    index.train(vectors[:2000])  # Train on sample
    index.add_batch(ids, vectors)  # Use add_batch!
    return index

def profile_search(index, queries, k=10, repeats=100):
    """Run multiple queries and measure average time."""
    start = time.perf_counter()
    all_results = []
    for _ in range(repeats):
        for query in queries:
            results = index.search(query, k=k)
            all_results.append(results)
    elapsed_s = time.perf_counter() - start
    mean_ms = (elapsed_s / (repeats * len(queries))) * 1000
    return all_results, mean_ms

def calculate_recall(flat_results, ivf_results, k=10):
    """Calculate what % of true top-k results IVF found."""
    total_recall = 0.0
    for flat_res, ivf_res in zip(flat_results, ivf_results):
        flat_ids = set(id_val for id_val, _ in flat_res[:k])
        ivf_ids = set(id_val for id_val, _ in ivf_res[:k])
        recall = len(flat_ids & ivf_ids) / k
        total_recall += recall
    return (total_recall / len(flat_results)) * 100

def run_benchmark(count=5000, dim=32, n_lists=50, k=10):
    print(f"\n{'='*60}")
    print(f"Benchmark: {count:,} vectors, {dim} dimensions, {n_lists} clusters")
    print(f"{'='*60}\n")
    
    # Generate data
    ids, vectors = make_dataset(count, dim)
    queries = vectors[:20]  # Use 20 queries for testing
    
    # Build indices
    print("Building FlatIndex...")
    flat = build_flat(ids, vectors, dim)
    
    print("Building IVF index...")
    ivf = build_ivf(ids, vectors, dim, n_lists)
    ivf_mem = memory_usage_mb()
    
    # Benchmark FlatIndex
    print("\nBenchmarking FlatIndex...")
    flat_all_results, flat_mean_ms = profile_search(flat, queries, k=k, repeats=50)
    flat_qps = 1000 / flat_mean_ms
    flat_mem = memory_usage_mb()
    
    print(f"\nFlatIndex Baseline:")
    print(f"  Latency: {flat_mean_ms:.3f} ms")
    print(f"  Throughput: {flat_qps:.1f} QPS")
    print(f"  Recall: 100.00% (exact search)")
    print(f"  Memory: {flat_mem:.1f} MB")
    
    # Benchmark IVF with different n_probe values
    print(f"\nIVF Results (n_lists={n_lists}):")
    print(f"  Memory: {ivf_mem:.1f} MB")
    print("-" * 60)
    
    for n_probe in [1, 2, 4, 8, 16]:
        if n_probe > n_lists:
            break
            
        # Update n_probe
        ivf.n_probe = n_probe
        
        # Benchmark
        ivf_all_results, ivf_mean_ms = profile_search(ivf, queries, k=k, repeats=50)
        
        # Calculate recall compared to flat results
        recall = calculate_recall(flat_all_results, ivf_all_results, k=k)
        
        speedup = flat_mean_ms / ivf_mean_ms
        ivf_qps = 1000 / ivf_mean_ms
        
        print(f"n_probe={n_probe:2d} | recall: {recall:5.1f}% | "
              f"latency: {ivf_mean_ms:6.3f}ms | "
              f"throughput: {ivf_qps:6.1f} QPS | "
              f"speedup: {speedup:4.1f}x")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Start small
    run_benchmark(count=5000, dim=8, n_lists=50, k=10)
    
    # Then scale up
    run_benchmark(count=10000, dim=8, n_lists=100, k=10)

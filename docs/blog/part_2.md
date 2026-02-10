# Building a Vector Search Library from Scratch: Part 2 - IVF (Inverted File Index)

## The Wall We Hit

In Part 1, we built a working vector search engine. It was correct, small, but completely impractical. Our `FlatIndex` compared every single vector in the database.
ALL 10M OF THEM! Taking over ~9 seconds per query.

The fundamental problem is that we are searching _everywhere_ when we should be searching _somewhere_.

Today, we are going to fix that.

## Analogy

Imagine you walk into a massive library looking for books about "Potato People of Plitenland". You don't check every single shelf in the building, instead, you go to the "History and Culture" section, and search ONLY in that. 

So you get your book in seconds instead of hours.

This is exactly what IVF does with vectors. We divide our space into neighbourhoods (clusters), and when a query comes in, we only search the neighbourhoods that are likely to contain similar vectors.

The trade off is that, we might miss the absolute best match if it happens to be in a cluster we did not search. But we got workarounds for that, believe me for now.

## Core Idea: Clustering

Try to think how vectors might naturally group together. Product descriptions for "running shoes" will cluster in one area of the vector space, while descriptions for "water bottles" cluster elsewhere.

If we can identify these groupings ahead of time, we can skip entire regions during search. A query about "pineapple" probably won't find its best matches in the "Bojack Horseman Merch" cluster.

Let's try to think of an algorithm.

Suppose we pick K random points as cluster centers (centroid), and assign each vector to its nearest centroid. Then we nudge the centroids to the average position of its assigned vectors. If we repeat this process (step 2 and 3) sufficient times, we will find that centroids will stop moving.

And that is what we want!

Let's see what this looks like in practice.

## Building the IVF Structure

We need to think about what an IVF index actually stores:

```py
from .base import BaseIndex

class IVFIndex(BaseIndex):
    def __init__(self, dim, metric="cosine", n_lists=100):
        super().__init__(dim, metric)
        self.n_lists = n_lists
        self.centroids = []
        self.inverted_lists = []
        self.inverted_ids = []
        self.is_trained = False
```

Did you notice something? Each cluster should be storing its own list of vectors and IDs. So shouldn't we write something like this?

```py
from .base import BaseIndex

class IVFIndex(BaseIndex):
    def __init__(self, dim, metric="cosine", n_lists=100):
        super().__init__(dim, metric)
        self.n_lists = n_lists

        # n_lists centroids, each is a vector
        self.centroids = []
        
        # vectors in each cluster
        self.inverted_lists = [[] for _ in range(n_lists)]
        
        # corresponding IDs
        self.inverted_ids = [[] for _ in range(n_lists)]
        self.is_trained = False
```

The term "inverted" comes from information retrieval. Normally, we map `vector -> data` but here, we are inverting it to `cluster -> vectors in cluster`.

It is like an index at the back of a book that maps topics to page numbers.

## Finding the Clusters

This is that point, where the paths diverge. Unlike `FlatIndex`, before we can add vectors, we need to TRAIN the index to discover the cluster structure. This requires a sample of vectors representative of our data.

Let's try to implement the algorithm which we conceptualised earlier.

```py
import random as r
# .. other code ..

def train(self, training_vectors):
    # we are picking random samples as centroids
    self.centroids = r.sample(training_vectors, self.n_lists)

    # nudges, im going with 25 iterations, which is a common choice for k-means
    for iters in range(25):

        # assigning vectors to the nearest centroid
        assignments = [[] for _ in range(self.n_lists)]
        for vec in training_vectors:
            nearest_index = self._find_nearest_centroid(vec)
            assignments[nearest_index].append(vec)

        # move centroids to the mean of their assigned vectors
        for i in range(self.n_lists):
            if assignments[i]:
                self.centroids[i] = self._compute_centroid(assignments[i])

    self.is_trained = True
```

We need to implement these two functions now: `_find_nearest_centroid()` and `_compute_centroid()`. If you had to remember one thing from this entire article, it would be the implementation of the former function stated above. That is literally the heartbeat of IVF. It will do the same thing as our search function, that is to find which cluster is closest to a vector. 

```py
from .kernels import SimilarityMetric
# .. other code ..

def _find_nearest_centroid(self, vector):
    best_score = float('-inf')
    best_index = 0

    q = vector

    if self.metric == "cosine":
        q = SimilarityMetric.normalize(vector)

    for i, centroid in enumerate(self.centroids):
        score = SimilarityMetric.score(self.metric, q, centroid)
        if score > best_score:
            best_score = score
            best_index = i

    return best_index
```

I have full faith on my audience that they can decipher what is being done with the code shown above.

```py
def _compute_mean(self, vectors):
    dim = len(vectors[0])
    mean = [0.0] * dim

    for vec in vectors:
        for i in range(dim):
            mean[i] += vec[i]

    for i in range(dim):
        mean[i] /= len(vectors)

    return mean
```

This is pretty straightforward as well.

## Adding Vectors to Clusters

Now that we can train the index, let's implement `add()`.

```py
def add(self, id, vector):
    if not self.is_trained:
        raise ValueError("CAN NOT ADD VECTORS BEFORE BEING TRAINED")

    v = vector
    if self.metric == "cosine":
        v = SimilarityMetric.normalize(vector)

    cluster_id = self._find_nearest_centroid(v)
    self.inverted_lists[cluster_id].append(v)
    self.inverted_ids[cluster_id].append(id)
```

Each vector gets assigned to exactly one cluster - the one whose centroid is closest. Over time, each inverted list fills up with similar vectors.

Well, with that, let's define the `__len__()` function now.

```py
def __len__(self):
    total = 0
    for inverted_list in self.inverted_lists:
        total += len(inverted_list)
    return total
```

## The Search 

Here is where IVF shows its power. Instead of searching all vectors, we find the closest cluster(s) to our quert, and then search only the vectors in those clusters. (Remember the library analogy?)

```py
import heapq
from .kernels import SimilarityMetric
# .. other code ..

def search(self, query_vector, k):
    if not self.is_trained:
        raise ValueError("MUST BE TRAINED BEFORE SEARCHING")

    q = query_vector
    if self.metric == "cosine":
        q = SimilarityMetric.normalize(query_vector)

    cluster_id = self._find_nearest_centroid(q)

    heap = []
    for pos in range(len(self.inverted_lists[cluster_id])):
        vec = self.inverted_lists[cluster_id][pos]
        id_val = self.inverted_ids[cluster_id][pos]
        score = SimilarityMetric.score(self.metric, q, vec)

        if len(heap) < k:
            heapq.heappush(heap, (score, id_val))
        else:
            heapq.heappushpop(heap, (score, id_val))
    
    return [(id_val, s) for s, id_val in sorted(heap, reverse=True)]    
```

Look at what we are doing here. If we have 100 clusters and our vectors are evenly distributed, we are searching ~1% of the database. That's a 100x speedup.

But hey, what if the best match is in the second closest cluster? We would miss it completely.

## Searching Multiple Clusters

We will introduce a parameter called `n_probe`. This is the key parameter for tuning the accuracy-speed trade off. We need to modify our `IVFIndex` class now.

```py
class IVFIndex(BaseIndex):
    def __init__(self, dim, metric="cosine", n_lists=100):
        super().__init__(dim, metric)
        # .. other code ..
        # how many clusters to search
        self.n_probe = n_probe
```

Now we need to fix our `search()` function to use `n_probe` and search the closest clusters.

```py
def search(self, query_vector, k):
    # .. other code ..

    # cluster_id = self._find_nearest_centroid(q)

    cluster_scores = []
    for i, centroid in enumerate(self.centroids):
        score = SimilarityMetric.score(self.metric, q, centroid)
        cluster_scores.append((score, i))

    cluster_scores.sort(reverse=True)
    clusters_to_search = [index for i, index in cluster_scores[:self.n_probe]]

    heap = []
    
    for cluster_id in clusters_to_search:
        # we just added a new loop around it
        for pos in range(len(self.inverted_lists[cluster_id])):
            vec = self.inverted_lists[cluster_id][pos]
            id_val = self.inverted_ids[cluster_id][pos]
            score = SimilarityMetric.score(self.metric, q, vec)

            if len(heap) < k:
                heapq.heappush(heap, (score, id_val))
            else:
                heapq.heappushpop(heap, (score, id_val))
    
    return [(id_val, s) for s, id_val in sorted(heap, reverse=True)]  
```

Now we kind of have a dial, where if we set `n_probe=1`, we wil have maximum speed, or `n_probe=10` (or more than that) for better accuracy.

With 100 clusters, `n_probe=10` means we search 10% of the database, that is still a 10x speedup, but with an okay-ish recall.

## Wiring it into the Engine

Be glad, that we had implemented this pattern beforehand.

```py
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
```

## Benchmarking: Does IVF actually deliver?

Time to see if all this theory translates into real performance gains. I ran benchmarks on relatively small datasets(5k and 10k vectors, 8 dimensions). These are intentionally modest, but are large enough to expose the core trade offs.

### Baseline: FlatIndex

First, we define the ground truth with brute force FlatIndex scan!


For 5000 vectors:

```
1. Mean Latency: ~3.65ms per query
2. Throughput: ~274 queries per second 
3. Recall: 100% (obviously)
```

For 10,000 vectors:

```
1. Mean Latency: ~7.45ms per query
2. Throughput: ~134 queries per second 
3. Recall: 100% (obviously)
```

This is our reference point. Latency scales linearly with dataset size, which was expected. For a single user clicking around that would feel almost instant, BUT if we try to handle 100 concurrent users, we would struggle to serve.

### IVF: The First Configuration 

#### n_lists=50 (5000 vectors)

Training builds 50 k-means centroids. After that we vary the `n_probe`:

```
n_probe=1  | recall 55.0% | latency 0.128 ms | 7806 QPS | 28.5x speedup
n_probe=2  | recall 72.0% | latency 0.211 ms | 4728 QPS | 17.3x speedup
n_probe=4  | recall 87.5% | latency 0.357 ms | 2802 QPS | 10.2x speedup
n_probe=8  | recall 98.5% | latency 0.646 ms | 1549 QPS |  5.7x speedup
n_probe=16 | recall 100%  | latency 1.316 ms |  760 QPS |  2.8x speedup
```

Look at the progression. 

At `n_probe=1`, IVF is extremely aggressive: nearly 30x faster, but recall collapses to 55%. That's rarely acceptable.

At `n_probe=4`, we hit a practical balance: 87.5% recall with a 10x speedup, cutting latency from 3.65 ms to 0.36 ms.

By `n_probe=8`, recall is effectively exact (98.5%) while still being almost 6x faster than brute force!

#### n_lists=100 (10,000 vectors)

Now we double both the dataset size and cluster count.

```
n_probe=1  | recall 67.5% | latency 0.168 ms | 5952 QPS | 44.3x speedup
n_probe=2  | recall 82.5% | latency 0.249 ms | 4015 QPS | 29.9x speedup
n_probe=4  | recall 92.0% | latency 0.407 ms | 2455 QPS | 18.3x speedup
n_probe=8  | recall 99.0% | latency 0.729 ms | 1371 QPS | 10.2x speedup
n_probe=16 | recall 100%  | latency 1.368 ms |  731 QPS |  5.4x speedup
```

Increasing `n_lists` improves both recall at low probes and overall speedups.

At `n_probe=4`, we get 92% recall with an 18x speedup. That is a strong default configuration with sub-millisecond latency and (okayish) high accuracy.

At `n_probe=8`, recall reaches 99% while still being 10x faster than brute force.

NOTE: Training is a one time cost, and even modest query volumes amortize training very quickly. After that point, every query is cheaper and faster.

## What the Numbers Tell Us

The numbers make the trade-off obvious and ... really exciting! On the 10k dataset, `n_probe=4` already hits ~93% recall while cutting latency from ~7.4 ms to ~0.4 ms and boosting throughput from ~136 QPS to ~2,400 QPS, which is an 18x (!!) speedup for only a small loss in accuracy. If we push to `n_probe=8` then recall jumps to ~99% while still staying about 11x faster than brute force. 

## What We're NOT Showing

These benchmarks use cosine similarity on randomly distributed vectors. Real-world data often clusters more naturally. Product embeddings, document vectors, image features tend to have structure. In those cases, IVF performs even better because the clusters are more coherent.

Conversely, if the vectors are truly uniformly distributed (like random noise), IVF struggles. The clusters don't capture meaningful patterns, and recall suffers.
You can try it on your own data to see how IVF performs.

## The Memory Elephant in the Room

So far, we have only talked about speed. Memory is the other half of the equation. IVF changes how we search, not what we store. That means IVF does not reduce memory usage by itself.

In a realistic system, vectors are stored as dense `float32` arrays. The memory cost is easy to compute (`memory per vector = dimensions * 4 bytes`), just take a look:

8D vector = 32 bytes  
128D vector = 512 bytes  

At scale, this will grow _really fast_.
100 million vectors at 128 dimensions will require around 48GB!

That is before accounting for metadata, indexes, or replicas. No matter how clever the indexing scheme is, storing raw embeddings becomes very expensive at large scale.

Solving this requires would changing the representation itself. Instead of storing every vector exactly, we need to somewhat compress them while preserving approximate distances.

And...this is where quantization enters the picture! 
The topic of Part 3.

## What's Next?

We are going to address memory. 
In Part 3 of this series, we will try to implement Product Quantization. 

The code is on Github, Benchmark scripts live inside the `benchmarks/` folder. Clone it and run experiments yourself!
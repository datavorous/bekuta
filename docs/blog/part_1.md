# Building a Vector Search Library from Scratch

Imagine that we are building a recommendation system. We have got 10M product descriptions, and a user just typed "running shoes for wide feet". How do you find the most relevant product in milliseconds?

This is where vector search comes in! Search queries, images, text etc. can all be represented as vectors (list of numbers). The challenge is simple: finding the most similar vectors as quickly as possible, when you have millions of them.

Let's build a toy vector search library from the ground up and discover why companies like Pinecone, Milvus, and FAISS need sophisticated techniques to handle billions of vectors.

## Starting Simple

The most straightforward way to find similar vectors is to compare them all. We need somewhere to store our vectors and a way to search through them.

So, let's define a container:

```py
class FlatIndex:
    def __init__(self, dim, metric):
        self.dim = dim
        self.metric = metric 
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}
```

The `vectors` list stores our data points, `ids` lets us label each vector, and `ids_to_index` helps us find vectors quickly by their ID.

### Understanding Similarity

Two vectors can be similar in different ways. Think of movie preferences between you and your friend. Both of you sci-fi (same direction), even if they've watched wayyy more movies (different magnitude). This is what cosine similarity will help us capture i.e. the angle between vectors regardless of what their length might be.

Mathematically we measure this using `a_vec (dot) b_vec = |a||b| cos theta`.

To make comparisons easier, we normalize vectors. After normalization, the dot product directly gives us the cosine similarity.

```py
def normalize(vector):
    mag_sq = 0
    for v in vector:
        mag_sq += v * v
    mag = mag_sq ** 0.5
    if mag == 0:
        return vector
    return [v / mag for v in vector]
```

Now we can add vectors to our index. For cosine similarity, we normalize during insertion so we do not have to do it repeatedly at query time:

```py
def add(index, id, vector):
    v = vector
    if v.index = "cosine":
        # we are normalizing here
        v = normalize(vector)

    pos = len(index.vectors)
    index.vectors.append(v)
    # id is given be the position of the vec in the list
    index.ids.append(id)
    index.ids_to_index[id] = pos
```

### The Common Metrics

Different use cases need different similarity measures:

1. Cosine Similarity: dot product of normalized vectors (measures directional alignment)
2. L2 Distance: euclidean distance (lower means more similar, so we negate it)

```py
def dot(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def score(metric, query, vector):

    if metric == "dot":
        return dot(query, vector)
    
    elif metric == "cosine":
        # we are making an assumption that the vectors are already normalized so we can just do a dot product
        return dot(query, vector)
    
    elif metric == "l2":
        # neg squared l2 distance because we want higher scores to be better
        dist_sq = 0
        for i in range(len(query)):
            diff = query[i] - vector[i]
            dist_sq += diff * diff
        return -dist_sq
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
```

### Our First Search Implementation

Our search strategy is quite straightforward. We will just compare the query against every vector, track the scores, and return the top-k results.

```py
def search(index, query_vector, k):
    q = query_vector
    if index.metric == "cosine":
        q = normalize(query_vector)

    results = []
    for pos in range(len(index.vectors)):
        id = index.ids[pos]
        s = score(index.metric, q, index.vectors[pos])
        results.append((id, s))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
```

Let's test this with three vectors where we can predict the outcome:

```py
from src.flat import FlatIndex, add, search

index = FlatIndex(dim=3, metric="cosine")

add(index, "vec1", [1, 0, 0])
add(index, "vec2", [0, 1, 0])
add(index, "vec3", [0.7, 0.7, 0])


query = [0.8, 0.6, 0]

results = search(index, query, k=2)

for id_val, score_val in results:
    print(f"id: {id_val}, score: {score_val}")
```

The output matches our intuition:

```bash
id: vec3, score: 0.9899494936611666
id: vec1, score: 0.8
```

Search took 0.015ms, with three vectors. Hmm, that's fair I guess?

### Finding the Breaking Point

How far can we push this simple approach? Let's scale up and watch what happens:

1. 10 vectors: 0.020ms, still instant  
2. 1000 vectors: 0.867ms, barely noticeable
3. 10,000 vectors: 8.077ms, getting slower OwO
4. 100,000 vectors: 72.808ms, now we are feeling it 
5. 10,000,000 vectors: 9256.012ms, lol.

Over 9 seconds for a single search, this won't work out well.

### Smarter Top-K Tracking

If you look closely, you'll notice something wasteful in our search. We sort all N items even though we only need the top-K results. When we have 10M vectors and only need the best 2, sorting everything is a huge overkill.

In this particular case, we can use a binary heap. Essentially, a heap is a tree where the parent is always smaller than it's children. For any index `i`, the left child sits at `2*i + 1`, the right child at `2*i + 2`, and the parent at `(i-1)//2`. 

When we maintain a heap of size k, we get O(N log k) complexity instead of O(N log N) from sorting. So when k << N, the difference gets massive.

```py
def heap_search(index, query_vector, k):
    import heapq

    q = query_vector
    if index.metric == "cosine":
        q = normalize(query_vector)

    heap = []
    for pos in range(len(index.vectors)):
        id = index.ids[pos]
        s = score(index.metric, q, index.vectors[pos])
        if len(heap) < k:
            heapq.heappush(heap, (s, id))
        else:
            heapq.heappushpop(heap, (s, id))

    return [(id, s) for s, id in sorted(heap, reverse=True)]
```

Let's spin up lots of vectors and see if we get any perf boost as the number of vectors grows:

```py
for i in range(10):
    add(index, f"vec{i}", [i % 10, (i // 10) % 10, (i // 100) % 10])
```

1. 1000 vectors: 1.269ms, the overhead hurts us here.
2. 10,000 vectors: 8.521ms, getting slower to our previous method
3. 100,000 vectors: 67.337ms, heap wins!
4. 10,000,000 vectors: 4532.593ms, nearly 2 times faster!

Nice.

### Some Nice-To-Have-s

Adding batch operations, deletions would be cool.

A naive batch insertion is straightforward:

```py
def add_batch(index, ids, vectors):
    for id, vector in zip(ids, vectors):
        add(index, id, vector)
```

and we would be able to use it like this:

```py
add_batch(index, ["a", "b", "c"], [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.1, 1.1, 1.1]
])
```

Deletion requires swapping the target with the last element to maintain contiguous arrays, and we can implement that like this:

```py
def delete(index, id):
    if id not in index.ids_to_index:
        return

    pos = index.ids_to_index[id]
    last_pos = len(index.vectors) - 1

    if pos != last_pos:
        index.vectors[pos] = index.vectors[last_pos]
        index.ids[pos] = index.ids[last_pos]
        index.ids_to_index[index.ids[pos]] = pos

    index.vectors.pop()
    index.ids.pop()
    del index.ids_to_index[id]
```

It should work.

```py
print("Before deletion:")
for id_val, vector in zip(index.ids, index.vectors):
    print(f"id: {id_val}, vector: {vector}")

# delete the index with id "b"
delete(index, "b")

print("After deletion:")
for id_val, vector in zip(index.ids, index.vectors):
    print(f"id: {id_val}, vector: {vector}")
```

And it does!

```
Before deletion:
id: a, vector: [1.0, 1.0, 1.0]
id: b, vector: [2.0, 2.0, 2.0]
id: c, vector: [1.1, 1.1, 1.1]
After deletion:
id: a, vector: [1.0, 1.0, 1.0]
id: c, vector: [1.1, 1.1, 1.1]
```

## Refactoring for Extensibility

Our code is tightly coupled to FlatIndex. Adding new index types like HNSW, IVF or LSH would require rewriting everything.

We need a common interface that all index types can follow. This way, the rest of the code doesn't need to know which specific index implementation is running underneath.

Let's define a base class with the core operations every index must support:

```py
class BaseIndex:
    def __init__(self, dim, metric="cosine"):
        self.dim = dim
        self.metric = metric

    def add(self, id, vector):
        raise NotImplementedError
    
    def add_batch(self, ids, vectors):
        for i in range(len(ids)):
            self.add(ids[i], vectors[i])

    def search(self, query_vector, k):
        raise NotImplementedError
    
    def delete(self, id):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
```

Let's wrap around our FlatIndex class around this.

```py
class FlatIndex(BaseIndex):
    def __init__(self, dim, metric="cosine"):
        super().__init__(dim, metric)
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}

    def add(self, id, vector):
        v = vector
        if self.metric == "cosine":
            v = SimilarityMetric.normalize(vector)

        pos = len(self.vectors)
        self.vectors.append(v)
        self.ids.append(id)
        self.ids_to_index[id] = pos

    def search(self, query_vector, k):
        q = query_vector
        if self.metric == "cosine":
            q = SimilarityMetric.normalize(query_vector)

        heap = []
        for pos in range(len(self.vectors)):
            id = self.ids[pos]
            s = SimilarityMetric.score(self.metric, q, self.vectors[pos])
            if len(heap) < k:
                heapq.heappush(heap, (s, id))
            else:
                heapq.heappushpop(heap, (s, id))

        return [(id, s) for s, id in sorted(heap, reverse=True)]

    def delete(self, id):
        if id not in self.ids_to_index:
            return

        pos = self.ids_to_index[id]
        last_pos = len(self.vectors) - 1

        if pos != last_pos:
            self.vectors[pos] = self.vectors[last_pos]
            self.ids[pos] = self.ids[last_pos]
            self.ids_to_index[self.ids[pos]] = pos

        self.vectors.pop()
        self.ids.pop()
        del self.ids_to_index[id]
        return True

    def __len__(self):
        return len(self.vectors)
```

How about we make the similarity computation independent such that any index can use it?
Wrapping around a class should be easy, it would be just copy pasting the the code of those functions inside the class.

```py
class SimilarityMetric:

    @staticmethod
    def dot(v1, v2):
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    @staticmethod
    def normalize(vector):
        mag_sq = 0
        for v in vector:
            mag_sq += v * v
        mag = mag_sq**0.5
        if mag == 0:
            return vector
        return [v / mag for v in vector]

    @staticmethod
    def score(metric, query, vector):
        if metric == "dot":
            return SimilarityMetric.dot(query, vector)

        elif metric == "cosine":
            return SimilarityMetric.dot(query, vector)

        elif metric == "l2":
            dist_sq = 0
            for i in range(len(query)):
                diff = query[i] - vector[i]
                dist_sq += diff * diff
            return -dist_sq

        else:
            raise ValueError(f"Unknown metric: {metric}")
```

And now it's time to update our FlatIndex class with this!

We will just replace orphan functions like `v = normalize(vector)` with `v = SimilarityMetric.normalize(vector)`.

Now it's quite obvious that we will go with the factory design pattern, it will help us avoid instantiating indexes directly.

### The Factory Pattern

The factory function prevents us from instantiating indexes directly and makes adding new types trivial.

```py
class Engine:
    @staticmethod
    def create_index(dim, metric="cosine", index_type="flat"):
        if index_type == "flat":
            return FlatIndex(dim, metric)
        raise ValueError(f"Unknown index type: {index_type}")
```

Now our usgae is clean and flexible.

```py
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
```

Say we wish to implement IVF, we would just need to inherit a class from `BaseIndex` and implement `add()`, `search()`, `delete()`, `__len__()`, and then add to the factory function. That's it!

## The Fundamental Problem

Our `FlatIndex` is exhaustive. To find the best match, it must examine every single vector in the database.

As we saw in our benchmarks, once we hit millions of vectors, the search jumps from milliseconds to seconds, and that is really bad.

So how do big players like pinecone, milvus and faiss handle billions of vectors with sub millisecond latency? Well, they stop looking at everything.

## What's Next

In Part 2, we will explore Approximate Nearest Neighbours (ANN) and implement Inverted File Index (IVF). Instead of comparing against every vector, we will use k-means clustering to partition our vector space into neighbourhoods. By only searching the neighbourhoods closest to our query, we can achieve massive speedups while maintaining good accuracy.

Until then, try pushing FlatIndex to its limits, and see how many vectors your machine can handle before it starts to cry :)

---

Note: I have packaged this as a proper Python library. You git clone my repo, follow the installation instructions and then import it anywhere:

```py
from bekuta import Engine
```

I also added a profiling utility that works as a decorator, just place it above any function you want to benchmark. 

Setting up the project structure is mechanical work that any AI coding assistant can help with, so I will skip those details here.
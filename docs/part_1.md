## building a vector search lib from scratch

we first need to understand what we actually need, and then figure out how it is generally implemented elsewhere, and how we might convert our understanding into code accordingly.

our aim is to find a way to store vectors (list of numbers) and then find the most similar ones to a given query vector.

for cases like this, the most common metrics would be to use cosine similarity (for direction), or L2 distance (for absolute distance) to compare vectors.

as we are just starting out, we will just compare the vectors one by one.

let us create a simple container first to hold our data.

```py
class FlatIndex:
    def __init__(self, dim, metric):
        self.dim = dim
        self.metric = metric 
        self.vectors = []
        self.ids = []
        self.ids_to_index = {}
```

`vectors` stores all our data points, `ids` lets us label each vector, and `ids_to_index` will help us to find vectors quickly by id.

now as we know, a_vec dot b_vec = |a||b|cos theta, and to make it convenient for us to compare the cosine similarity, we could just normalize the vectors; essentially dividing each component of the vector using its magnitude.

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

now with these done, we should move to implement something to add vectors to the index.

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

now we need to implement, i.e. computing the similarity scores

there are different ways to measure how similar two vectors are. we could use 

dot products: sum of element wise prod (higher is more similar)

cosine similarity: dot product of normalized vectors

l2 dist: euclidean distance (lower is more similar, but we will negate it)

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

let's implement the most basic search (brute force)

our aim is to find the k most similar vectors to a given query

what we may do is, compare the query to every vector, and keep track of the best k results. hence lets use a simple list to track top-k, and then sort at the end

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

let's us try to check if our minimal implementation works or not.

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

the output we get:

```bash
id: vec3, score: 0.9899494936611666
id: vec1, score: 0.8
```

it took 0.015ms for the search operation to take place

notice, that we are sorting all results even when we only need top-k, 

can there be any data structure which would help us keep the smallest item at the top and can allow to efficiently add/remove items? 

that sounds like a job for binary heap.

essentiallly it is a tree, where parent is always smaller than the children.

we store items in a list, where index 0 is the root (minimum)
now for any item at index i, we would 

get the left child at index `2*i + 1`, and right at `2*i + 2`, and parent would be in `((i-1)//2)`

while adding, we put that value at the end, then "bubble up" by swapping with parent if smaller.

and when removing we take the root, move the last item to root, then "bubble down" with smallest child 

let's implement it using python's `heapq`.

we will make another function for now, to help us in benchmarking

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

on testing our previous example, we find that the time is much higher than naive implemention 

```bash
Search took 0.015ms
Heap search took 0.379ms
id: vec3, score: 0.9899494936611666
id: vec1, score: 0.8
```

this is because we are using just 3 vectors, and the overhead is not being amortized

let's spin up lots of vectors and see if we get any perf boost as the number of vectors grows or not.

```py
for i in range(10):
    add(index, f"vec{i}", [i % 10, (i // 10) % 10, (i // 100) % 10])
```

with 10:
Search took 0.020ms
Heap search took 0.323ms
id: vec1, score: 0.8
id: vec2, score: 0.8

nope

1000:
Search took 0.867ms
Heap search took 1.269ms
id: vec34, score: 1.0
id: vec68, score: 1.0

still no...

10,000

Search took 8.077ms
Heap search took 8.521ms
id: vec34, score: 1.0
id: vec68, score: 1.0

we are getting close..

100,000:

Search took 72.808ms
Heap search took 67.337ms
id: vec34, score: 1.0
id: vec68, score: 1.0

ah here we go!

10,000,000?

Search took 9256.012ms
Heap search took 4532.593ms
id: vec34, score: 1.0
id: vec68, score: 1.0

damn

instead of sorting N items O(N log N), we maintain a heap of size k O(N log K), which is much faster when k << N

what can we do next to make this a usable artifact?

uh maybe adding a function to delete vectors, and having batch implementation?

batch insertion is easy tbf:

```py
def add_batch(index, ids, vectors):
    for id, vector in zip(ids, vectors):
        add(index, id, vector)
```
and we got that working like that

```py
add_batch(index, ["a", "b", "c"], [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.1, 1.1, 1.1]
])
```
let's implement the deleting part

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

output:

```
Before deletion:
id: a, vector: [1.0, 1.0, 1.0]
id: b, vector: [2.0, 2.0, 2.0]
id: c, vector: [1.1, 1.1, 1.1]
After deletion:
id: a, vector: [1.0, 1.0, 1.0]
id: c, vector: [1.1, 1.1, 1.1]
```

well this looks usable as a toy 

can we restructure it in any better way?

right now our code is tightly coupled to FlatIndex, if we wanted to add new index types (like HNSW, IVF or LSH), we would have to rewrite a lot of code, so we need to refactor to make adding new indexes easier

we would need a common interface that all index types follow, so the rest of our code dont care doesnt care which index is being used

we might use an abstract class or protocol to define the interface, then go on to implement concrete index types

so let's create a base class that defines what methods every index must have, then refact FlatIndex to inherit from it

something like this should work for us:

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

let's wrap around our FlatIndex around this

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
                v = normalize(vector)

            pos = len(self.vectors)
            self.vectors.append(v)
            self.ids.append(id)
            self.ids_to_index[id] = pos

        def search(self, query_vector, k):
            q = query_vector
            if self.metric == "cosine":
                q = normalize(query_vector)

            heap = []
            for pos in range(len(self.vectors)):
                id = self.ids[pos]
                s = score(self.metric, q, self.vectors[pos])
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

how about we make the similarity computation independent such that any index can use it?
wrapping around a class should be easy, its just copy pasting the the code of those functions inside the class

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

add now time to update our FlatIndex class with this!

it is just replacing orphan functions like 

```py
v = normalize(vector)
```

with

```py
v = SimilarityMetric.normalize(vector)
```

now it's quite obvious that we will go with the factory design principle, it will help us avoid instantiating indexes directly

```py
class Engine:
    def create_index(dim, metric="cosine", index_type="flat"):
        if index_type == "flat":
            return FlatIndex(dim, metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
```

that should allow us to plug more indexes easily.

now we can just do things like this:

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


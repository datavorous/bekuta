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

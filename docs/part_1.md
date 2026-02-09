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

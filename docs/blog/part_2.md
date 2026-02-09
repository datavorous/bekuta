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
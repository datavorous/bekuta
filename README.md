<p align="center">
  <img src="media/logo.png" alt="bekuta logo" width="200">
</p>


<p align="center">
    <b>bekuta</b><br>
    A toy vector search library written in Python.
</p>

Install:
```bash
git clone https://github.com/datavorous/bekuta.git
cd bekuta
uv pip install -e .
```

Usage:
```python
from bekuta import Engine

# FlatIndex (brute-force)
index = Engine.create_index(dim=3, metric="l2", index_type="flat")
index.add_batch(["a", "b"], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
print(index.search([0.9, 0.9, 0.9], k=1))

# IVFIndex (inverted file with clustering)
ivf = Engine.create_index(dim=3, metric="l2", index_type="ivf", n_lists=100, n_probe=4)
ivf.train([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])  # training required
ivf.add("a", [1.0, 1.0, 1.0])
ivf.add("b", [2.0, 2.0, 2.0])
print(ivf.search([0.9, 0.9, 0.9], k=1))
```

**Indexes:** 
- FlatIndex (brute-force with max heap for top-k)
- IVFIndex (inverted file with Voronoi clustering, requires training)

**Metrics:** cosine, dot, l2

**Benchmarks:** Performance comparison scripts are available in `benchmarks/` directory.

Profile functions with `@profile` decorator:
```python
from profiling import profile

@profile
def my_function():
    pass
```

Outputs: `[PROFILE] my_function took X.XXXms`

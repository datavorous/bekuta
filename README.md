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

index = Engine.create_index(dim=3, metric="l2", index_type="flat")
index.add_batch(["a", "b"], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
print(index.search([0.9, 0.9, 0.9], k=1))
```

**Indexes:** FlatIndex (brute-force with max heap for top-k)

**Metrics:** cosine, dot, l2

Profile functions with `@profile` decorator:
```python
from profiling import profile

@profile
def my_function():
    pass
```

Outputs: `[PROFILE] my_function took X.XXXms`

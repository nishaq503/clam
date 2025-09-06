# Reproducible results for the CAKES paper

# Usage

```sh
uv run paper-cakes --help
```

# Recurrence Relations

Start by generating some data, e.g. with `max_n` = 100k and `max_k` = 16

```sh
uv run paper-cakes recurrence-relations gen-ratios -d ./papers/cakes/data/recurrence_relations -n 100000 -k 16
```

Start the jupyter notebook called `recurrence_relations.ipynb`.

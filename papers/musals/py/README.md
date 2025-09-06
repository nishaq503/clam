# Reproducible results for the Musals paper

# Usage

```sh
uv run paper-musals --help
```

# Recurrence Relations

Start by generating some data, e.g. with `max_n` = 100k and `max_k` = 16

```sh
uv run paper-musals recurrence-relations gen-ratios -d ./papers/musals/data/recurrence_relations -n 100000 -k 16
```

Start the jupyter notebook called `recurrence_relations.ipynb`.

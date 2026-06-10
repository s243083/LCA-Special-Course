# Contributing

## Building the docs locally

```bash
pip install -e ".[docs]"
cd docs
make html          # Linux/macOS
# or: make.bat html     (Windows)
```

Output: `docs/_build/html/index.html`.

## Docstring style

Use NumPy-style docstrings (Parameters / Returns / Examples blocks) so
`sphinx.ext.napoleon` renders them cleanly in the API reference.

## Adding a tutorial

Drop a new `.ipynb` under `docs/tutorials/` (or link one from
`examples/`) and reference it in `docs/tutorials/index.md`.

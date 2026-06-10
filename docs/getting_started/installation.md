# Installation

WINPACT requires **Python ≥ 3.11**.

## Using conda / mamba (recommended)

```bash
mamba env create -f environment.yml
mamba activate winpact
pip install -e .
```

## Using pip

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

## Optional extras

| Extra       | Purpose                                         |
|-------------|-------------------------------------------------|
| `notebook`  | Run tutorial notebooks (`ipykernel`, `nbformat`) |
| `internal`  | Private DTU dependencies (e.g. `maesopt`)        |
| `docs`      | Build this documentation locally                 |

```bash
pip install -e ".[notebook,docs]"
```

## Verifying the install

```python
import winpact
print(winpact.__version__)
```

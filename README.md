# WINPACT

📚 **Documentation:** <https://hipersim.pages.windenergy.dtu.dk/winpact/>

WINPACT is a Python-based life-cycle modeling tool for impact assessment of wind energy technologies. It simulates the operational life of wind energy assets and their interaction with metocean, technical, and economic environments. The tool quantifies the economic and technical effects of system changes, including wind farm control strategies, turbine design characteristics, external operating conditions, and economic parameters.

WINPACT captures combined system interactions such as control-driven load changes, reliability impacts, O&M cost evolution, market interaction, and revenue potential. It provides aggregated outputs in terms of economic metrics (e.g., LCOE, NPV) and technical metrics (e.g., component, turbine, and farm health).

The tool is developed and maintained as a Python software chain by the Integrity and Reliability Section at DTU Wind and Energy Systems.




## 🪪 License

This software — **WINPACT** — is licensed under the [Apache License 2.0](LICENSE).

© 2025 Integrity and Reliability (IAR) Section, Technical University of Denmark (DTU).

You are free to use, modify, and distribute this software (including in commercial or proprietary projects) under the conditions of the Apache 2.0 license.

See the accompanying [NOTICE](NOTICE) file for attribution details.

---

## 📖 How to Cite

If you use **WINPACT** in your research or project, please cite it as follows:

Gräfe, M., Pettas, V., Ioannou, A., Shields, M., Kolios, A., & Dimitrov, N. (2026). WINPACT. Zenodo. https://doi.org/10.5281/zenodo.19070012

---

# Installation


WINPACT is a Python package and can be installed into an environment that provides a compatible Python version.

## Requirements

* **Python ≥ 3.11**
* `pip` (recent version recommended)
* (Optional but recommended) Conda or Mamba for environment management

> ⚠️ `pip` does **not** create or manage Python environments. You must activate an environment with a compatible Python version **before** installing WINPACT.

---

### 1. Create and activate an environment

#### Option A: Conda / Mamba (recommended)

```bash
conda create -n winpact python=3.11
conda activate winpact
```

#### Option B: Python venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\\Scripts\\activate       # Windows
```

---

### 2. Clone the repository

```bash
git clone https://gitlab.windenergy.dtu.dk/HiperSim/winpact.git
cd winpact
```

The repository root must contain `pyproject.toml`.



### 3. Install WINPACT

#### Base install (library only)

```bash
pip install .
```

Installs WINPACT and its required dependencies.

---

#### Editable install (recommended for development and notebooks)

```bash
pip install -e .
```

An editable install links the installed package directly to the cloned repository:

* Code changes take effect immediately
* No reinstallation is required after edits
* Ideal for development and running notebooks in `winpact/examples`

---

### 4. Optional installation extras

#### Notebook support

```bash
pip install -e ".[notebook]"
```

Installs additional dependencies required for running example notebooks (e.g. `ipykernel`, `nbformat`).

---

#### Internal dependencies (Needed for TREND UseCase, BladeMassEstimator)

```bash
pip install -e ".[internal]"
```

Installs optional internal dependencies from private Git repositories. This requires appropriate access rights.

Extras can be combined:

```bash
pip install -e ".[notebook,internal]"
```

---

## 📚 Documentation

The full WINPACT documentation — installation, tutorials, theory, and the
auto-generated API reference — is published at:

**<https://hipersim.pages.windenergy.dtu.dk/winpact/>**

To build the docs locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

---

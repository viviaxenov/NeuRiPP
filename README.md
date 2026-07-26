# NeuRiPP
**Ri**emannian methods for **Neu**ral **P**ushforward distributions with **P**ullback Wasserstein metric

## Installation
NeuRiPP requires Python 3.12 or newer. The default installation uses the CPU
backend; CUDA 12 and CUDA 13 support are available as optional targets.

Install the core package directly from GitHub:

```bash
python -m pip install "NeuRiPP @ git+https://github.com/viviaxenov/NeuRiPP.git"
```

Select a CUDA target or install the dependencies used by the examples with pip
extras:

```bash
python -m pip install "NeuRiPP[cuda12] @ git+https://github.com/viviaxenov/NeuRiPP.git"
python -m pip install "NeuRiPP[cuda13] @ git+https://github.com/viviaxenov/NeuRiPP.git"
python -m pip install "NeuRiPP[cuda12,examples] @ git+https://github.com/viviaxenov/NeuRiPP.git"
```

The CUDA extras install the CUDA runtime libraries distributed through pip. Do
not install the `cuda12` and `cuda13` extras together.

### Editable installation

Clone the repository and select the required target. For a complete CUDA 12
development installation with all example dependencies:

```bash
git clone git@github.com:viviaxenov/NeuRiPP.git
cd NeuRiPP
python -m pip install -e ".[cuda12,examples]"
```

Other supported combinations are:

```bash
python -m pip install -e .                         # CPU, core only
python -m pip install -e ".[examples]"             # CPU with examples
python -m pip install -e ".[cuda12]"               # CUDA 12, core only
python -m pip install -e ".[cuda13]"               # CUDA 13, core only
python -m pip install -e ".[cuda13,examples]"      # CUDA 13 with examples
python -m pip install -e ".[cuda12,examples,docs]" # CUDA 12, examples and docs
```

The supported stack pins JAX 0.11.0 and Flax 0.12.8. The `examples` extra also
installs `uncprop` from the configured GitHub fork. A fresh virtual environment
is recommended when switching CPU or CUDA targets.

### Building docs
[See](./docs/README.md)

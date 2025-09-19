# NeuRiPP
**Ri**emannian methods for **Neu**ral **P**ushforward distributions with **P**ullback Wasserstein metric

## Installation
[Optional] Install dependencies with `conda`
```
    conda install --file requirements_conda.txt
```
Install the package
```
    pip install NeuRiPP@git+https://github.com/viviaxenov/NeuRiPP
```
Editable (developer) installation
```
    git clone git@github.com:viviaxenov/NeuRiPP.git
    cd NeuRiPP
    conda install --file requirements_conda.txt
    pip install -e .
```

### Requirements
 - `flax>=0.10.2`
 - `jax>=0.6.2`
 - `jaxtyping>=0.3.2`
 - `matplotlib>=3.10.6`
 - `numpy`
 - `optax>=0.2.6`
 - `tqdm>=4.67.1`
 - `typing_extensions>=4.15.0`

### Building docs
[See](./docs/README.md)

# Installation
We have two different configs for DDRNetOC (since it requires older wersion of PyTorch) and all other models.

## Environment Setup

### Base config
- Ubuntu 20.04.3 LTS
- Python 3.8
- [PyTorch v1.8.0 (CUDA 11.1 build)](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.9.0](https://pytorch.org/get-started/previous-versions/)

Create a conda environment.
  
```bash
conda create --name daps_base --file requirements_base.txt
conda activate daps_base
```

### DDRNetOC config
- Ubuntu 20.04.3 LTS
- Python 3.7
- [PyTorch v1.1.0 (CUDA 10.0 build)](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.2.2](https://pytorch.org/get-started/previous-versions/)

Create a conda environment.
  
```bash
conda create --name daps_oc --file requirements_base.txt
conda activate daps_oc
```
## WandB Setup

Use WandB in order to track your runs.

Create a login a file in the project directory `DAPS3D/wandb.login`:

```bash
Experiments_daps3d    <----    <project name>
subake                <----    <account name>
key                   <----    <wandb login key>
```

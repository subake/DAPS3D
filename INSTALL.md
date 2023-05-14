# Installation
We have two different configs for DDRNetOC (since it requires older wersion of PyTorch) and for all other models.

## Environment Setup

### Base config
- Ubuntu 20.04.3 LTS
- Python 3.8
- [PyTorch v1.8.0 (CUDA 11.1 build)](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow v2.2.0](https://www.tensorflow.org/install)

Create a conda environment.
  
```bash
conda create -n daps_base python=3.8
conda activate daps_base
```
Install PyTorch, TensorFlow and other dependencies.

```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements_base.txt
```

### DDRNetOC config
- Ubuntu 20.04.3 LTS
- Python 3.7
- [PyTorch v1.1.0 (CUDA 10.0 build)](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow v1.13.1](https://www.tensorflow.org/install)

Create a conda environment.
  
```bash
conda create -n daps_oc python=3.7
conda activate daps_oc
```

Install PyTorch, TensorFlow and other dependencies.

```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge cudatoolkit-dev

pip install -r requirements_oc.txt
```

## WandB Setup

Use WandB in order to track your runs.

Create a login a file in the project directory `DAPS3D/wandb.login`:

```bash
Experiments_daps3d    <----    <project name>
subake                <----    <account name>
key                   <----    <wandb login key>
```

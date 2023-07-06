# Multimodal self-supervised learning and network for 3D-to-2D segmentation


**Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation**, José Morano, Guilherme Aresta, Dmitrii Lachinov, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović. Accepted at MICCAI 2023. Available at [arXiv](...).

## Approach

![diagrams-Approach](https://github.com/j-morano/SSL-3D-to-2D/assets/48717183/31aa21fc-6a0f-41ad-b410-7d1aaefca79a)

## Network architecture

![diagrams-Architecture](https://github.com/j-morano/SSL-3D-to-2D/assets/48717183/40fb6b7c-2479-46e7-aa21-4dc180967e10)


## Setting up the environment

The code should work with the latest versions of Python (3.11.4) and PyTorch (2.0.1) and with CUDA 11.7.

The recommended way to set up the environment is using a Python virtual environment.

To do so, you can run the following commands:

```shell
# Create Python environment
python3 -m venv venv

# Activate Python environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```


## Setting up the _original_ environment

All the experiments were originally run in a server with Python 3.6.8, PyTorch 1.10.2, and CUDA 11.3.

To install this version of Python, you can use [pyenv](https://github.com/pyenv/pyenv), which can be easily installed using [pyenv-installer](https://github.com/pyenv/pyenv-installer).

Moreover, the original requirements are listed in `original-requirements.txt`.

So, to set up the original environment, you can run the following commands:

```shell
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.6.8
pyenv install -v 3.6.8

# Create Python environment
$PYENV_ROOT/versions/3.6.8/bin/python3 -m venv venv

# Activate Python environment
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r original-requirements.txt
```


## Run training

See `run.sh`.

Available options can be found in `config.py`.

Our proposed 3D-to-2D segmentation network, `FPN`, is available in `models/fusion_nets.py`.


## Citation

If you find this repository useful in your research, please cite:

```
@article{morano2021self,
  title={Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation},
  author={Morano, Jos{\'e} and Aresta, Guilherme and Lachinov, Dmitrii and Mai, Julia and Schmidt-Erfurth, Ursula and Bogunovi{\'c}, Hrvoje},
  journal={arXiv preprint arXiv:2106.16071},
  year={2023}
}
```

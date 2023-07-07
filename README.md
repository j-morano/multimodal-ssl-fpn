# Multimodal self-supervised learning and network for 3D-to-2D segmentation


**Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation**, José Morano, Guilherme Aresta, Dmitrii Lachinov, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović. Accepted at MICCAI 2023. Available at [arXiv](https://doi.org/10.48550/arXiv.2307.03008).

## Approach

![image](https://github.com/j-morano/multimodal-ssl-fpn/assets/48717183/34baf433-4563-446f-be22-8c7f1c36c340)


## Network architecture

![image](https://github.com/j-morano/multimodal-ssl-fpn/assets/48717183/511a9e95-994c-47ea-9e7f-9a2f17a1a1aa)


Our proposed 3D-to-2D segmentation network, `FPN`, is available in `models/fusion_nets.py`.

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


## Citation

If you find this repository useful in your research, please cite:

```
@misc{morano2023selfsupervised,
      title={Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation}, 
      author={José Morano and Guilherme Aresta and Dmitrii Lachinov and Julia Mai and Ursula Schmidt-Erfurth and Hrvoje Bogunović},
      year={2023},
      eprint={2307.03008},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      doi={10.48550/arXiv.2307.03008}
}
```

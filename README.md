# Multimodal self-supervised learning approach and network for 3D-to-2D tasks

This repository contains the source code of the following article:

**Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation**, José Morano, Guilherme Aresta, Dmitrii Lachinov, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović. To appear in MICCAI 2023. Available at [arXiv](https://doi.org/10.48550/arXiv.2307.03008).

## Approach

![image](https://github.com/j-morano/multimodal-ssl-fpn/assets/48717183/c6a9b8e6-66c8-4fbe-9f59-099e9e3bb0a4)


## Network architecture

![image](https://github.com/j-morano/multimodal-ssl-fpn/assets/48717183/26f9b65d-aeeb-4d9f-a82b-80e92b3a77ae)


Our proposed 3D-to-2D segmentation network, `FPN`, is available in `models/fusion_nets.py`, along with the implementation of state-of-the-art networks Lachinov et al. ([MICCAI 2021](https://doi.org/10.48550/arXiv.2108.00831)) and ReSensNet ([Seeböck et al., Ophthalmology Retina, 2022](https://doi.org/10.1016/j.oret.2022.01.021)).

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
      author={Jos{\'{e}} Morano and Guilherme Aresta and Dmitrii Lachinov and Julia Mai and Ursula Schmidt-Erfurth and Hrvoje Bogunovi{\'{c}}},
      title={Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation},
      publisher={arXiv},
      year={2023},
      doi={10.48550/arXiv.2307.03008}
}
```

Moreover, if you use any of the state-of-the-art networks, please cite the corresponding paper:

```
@misc{lachinov2021projective,
  author = {Dmitrii Lachinov and Philipp Seeb\"{o}ck and Julia Mai and Ursula Schmidt-Erfurth and Hrvoje Bogunovi{\'{c}}},
  title = {Projective Skip-Connections for Segmentation Along a Subset of Dimensions in Retinal OCT},
  publisher = {arXiv},
  year = {2021},
  doi = {10.48550/ARXIV.2108.00831}
}
```

```
@article{seebock2022linking,
  author = {Philipp Seeb\"{o}ck and Wolf-Dieter Vogl and Sebastian M. Waldstein and Jose Ignacio Orlando and Magdalena Baratsits and Thomas Alten and Mustafa Arikan and Georgios Mylonas and Hrvoje Bogunovi{\'{c}} and Ursula Schmidt-Erfurth},
  title = {Linking Function and Structure with {ReSensNet}},
  journal = {Ophthalmology Retina},
  doi = {10.1016/j.oret.2022.01.021},
  year = {2022},
  month = jun,
  publisher = {Elsevier {BV}},
  volume = {6},
  number = {6},
  pages = {501--511}
}
```

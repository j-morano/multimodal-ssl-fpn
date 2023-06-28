# SSL for 3D-to-2D segmentation


**Self-supervised learning via inter-modal reconstruction and feature projection networks for label-efficient 3D-to-2D segmentation**, José Morano, Guilherme Aresta, Dmitrii Lachinov, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović. Accepted at MICCAI 2023. Available at [arXiv](...).

## Approach


## Architecture


## Setting up the environment

Create and activate Python environment. Install requirements using `requirements.txt`.

```shell
python3 -m venv venv/
source venv/bin/activate
pip3 install -r requirements.txt
```

## Run training

See `run.sh`.

Available options can be found in `config.py`.


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


## Environment used for the experiments

* Python 3.6.8
* PyTorch 1.10.2+cu113

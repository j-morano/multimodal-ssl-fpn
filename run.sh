#!/bin/bash

BASE_DIR=/run/media/morano/SW1000/OPTIMA/Source/src/GitHub_multimodal-ssl-fpn

source $BASE_DIR/venv/bin/activate
cd $BASE_DIR


python3 --version

python3 train.py --model FPN --dataset InvertedFAFReconstruction --data-ratio 0.05

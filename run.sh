#!/bin/bash

BASE_DIR=/path/to/multimodal-ssl-fpn

source $BASE_DIR/venv/bin/activate
cd $BASE_DIR


python3 --version

python3 train.py --model FPN --dataset InvertedFAFReconstruction --data-ratio 0.05

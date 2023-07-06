from typing import Union
import argparse
import socket



parser = argparse.ArgumentParser()
# Data options
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data-ratio", type=float, default=1.0)
# Execution options
parser.add_argument("--exec-test", action='store_true', help="execution test")
parser.add_argument("--version", type=str, default="v0")
parser.add_argument("--gpus", type=Union[list, int], default=1)
parser.add_argument("--cache-strategy", type=str, default="DoNotReleaseMemCache")
parser.add_argument("--threads", type=int, default=8)
# Training hyperparameters
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--virtual-batch-size", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-1)
# Model options
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model-weights", type=str, default=None)
parser.add_argument("--num-outputs", type=int, default=1)
parser.add_argument("--multiplier", type=int, default=20)
config = parser.parse_args()


# Extra options
config.models_path = f'./__train/{config.version}/'

# Overwrite some configs when running in local machine
if socket.gethostname() in ['hemingway']:
    config.batch_size = 1
    config.gpus = [0]
    config.virtual_batch_size = 1
    config.threads = 1
    config.cache_strategy = "ReleaseMemCache"
    config.multiplier = 1

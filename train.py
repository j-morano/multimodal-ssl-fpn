#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import random
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import config
import pl_model_wrapper
from models import weight_init
from models.fusion_nets import model_factory
from data.data_config import data_config_factory



data_config = data_config_factory[config.dataset]()


def worker_init_fn(worker_id):
    seed = torch.initial_seed() + worker_id
    np.random.seed([int(seed%0x80000000), int(seed//0x80000000)]) # type: ignore
    torch.manual_seed(seed)
    random.seed(seed)


def main(model_path, training_file_list=None, validation_file_list=None):
    pl.seed_everything(1234)

    if training_file_list is None or validation_file_list is None:
        print('The training or validation list is empty')

    print("===> Building model")
    arch = model_factory[config.model]()
    if config.model_weights is None:
        print('Random initialization')
        arch.apply(weight_init.weight_init)

    print("===> Loading datasets")
    print('Train data:', data_config.paths['data'])

    print('Train:', training_file_list)
    print('Val:', validation_file_list)

    data_transform, data_transform_val = data_config.get_transforms()

    train_data = data_config.train_data(
        training_file_list,
        data_transform,
        config.multiplier
    )
    val_data = data_config.val_data(validation_file_list, data_transform_val)

    training_data_loader = DataLoader(
        dataset=train_data,
        num_workers=config.threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    evaluation_data_loader = DataLoader(
        dataset=val_data,
        num_workers=config.threads,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    criterion = data_config.get_criterion()

    metrics_train = data_config.metrics_train
    metrics_val = data_config.metrics_val

    meta_metric_val = data_config.meta_metric_val

    # PL callback to store top-5 models (Dice)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='{epoch}-{'+data_config.monitor+':.4f}',
        save_top_k=5,
        # verbose=True,
        monitor=data_config.monitor,
        mode='max',
        # prefix=''
        save_weights_only=True,
    )

    optimizers = [
        torch.optim.SGD(
            arch.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        ),
    ]

    compiled_model = pl_model_wrapper.Model(
        model=arch,
        losses=criterion,
        training_metrics=metrics_train,
        metrics=metrics_val,
        metametrics=meta_metric_val,
        optim=(optimizers, []),
        force_mem_cache_release=config.cache_strategy,
        model_path=model_path,
    )

    if config.model_weights is not None:
        print('Loading pretrained model')
        checkpoint = torch.load(config.model_weights)
        try:
            compiled_model.load_state_dict(checkpoint['state_dict'], strict=True)
        except KeyError:
            compiled_model.load_state_dict(checkpoint, strict=True)

    trainer = pl.Trainer(
        logger=False,
        callbacks=[checkpoint_callback],
        # log_every_n_steps=10,
        precision=32,
        devices=config.gpus,
        num_sanity_val_steps=2,
        accumulate_grad_batches=config.virtual_batch_size,
        max_epochs=config.epochs,
        sync_batchnorm=False,
        benchmark=True,
        accelerator='gpu',
        # strategy='ddp',
    )

    if config.exec_test:
        print(arch)
        print('Testing mode enabled. Skipping training.')
        return

    print("===> Begin training")
    trainer.fit(
        compiled_model,
        train_dataloaders=training_data_loader,
        val_dataloaders=evaluation_data_loader
    )

    if trainer.state.status == 'interrupted':
        print('Training interrupted')
    else:
        print("===> Saving last model")
        trainer.save_checkpoint(os.path.join(model_path, 'last.ckpt'), weights_only=True)

    

def train_with_split(split):
    # Build model path and create dir if it does not exist
    model_path = os.path.join(
        config.models_path,
        config.dataset,
        Path(data_config.paths['split']).stem,
        str(config.data_ratio),
    )
    model_name = config.model
    if config.model_weights is not None:
        model_name = model_name + '-' + Path(config.model_weights).stem
    model_path = join(model_path, model_name)

    Path(model_path).mkdir(exist_ok=True, parents=True)
    print(model_path)

    # Copy run files
    shutil.copy2('./run.sh', model_path)

    # Folder to store monitoring images
    Path(os.path.join(model_path, 'images')).mkdir(exist_ok=True, parents=True)

    train_ids, val_ids = split['train'], split['val']

    if config.data_ratio < 1.0:
        print('Using only', config.data_ratio*100, '% of the training data.')
        limit = min(1, int(len(train_ids)*config.data_ratio))
        train_ids = train_ids[:limit]
        ## val_ids = val_ids[:limit]

    print('Train:', train_ids, '\nVal:', val_ids)
    print('Number of training samples:', len(train_ids))
    print('Number of validation samples:', len(val_ids))

    main(model_path, train_ids, val_ids)


if __name__ == "__main__":
    split = data_config.paths['split']
    with open(split, 'r') as fp:
        splits = json.load(fp)

    print('Using split:', Path(split).stem)

    train_with_split(splits)

from os.path import join
import gc
import time
import json
from typing import Optional

import torch
import pytorch_lightning as pl
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.morphology import binary_erosion
from skimage.morphology import disk
import time
import matplotlib
from matplotlib import pyplot as plt

from factory_utils import get_factory_adder



matplotlib.use('Agg')
pyplot_colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


add_cache_strategy, mem_cache_strategies = get_factory_adder()


@add_cache_strategy
class ReleaseMemCache:
    def __call__(self):
        # Force garbage collection and cache memory freeing
        gc.collect()
        torch.cuda.empty_cache()


@add_cache_strategy
class DoNotReleaseMemCache:
    def __call__(self):
        pass


class MonitorLearning:
    def __init__(self):
        self.minute = -1

    def is_save_time(self):
        # Save one image for debugging every minute
        now_minute = int(time.time() / 60)
        is_save_time = now_minute > self.minute
        if is_save_time:
            self.minute = now_minute
        return is_save_time


class Model(pl.LightningModule):

    def __init__(
        self,
        model,
        losses,
        training_metrics,
        metrics,
        metametrics,
        optim,
        force_mem_cache_release="DoNotReleaseMemCache",
        validation: Optional[str]=None,
        model_path: Optional[str]=None,
    ):
        super().__init__()
        self.model = model
        self.loss = losses
        self.metrics = metrics
        self.metametrics = metametrics
        self.optim = optim
        self.training_metrics = training_metrics
        self.validation = validation
        print(self.validation)
        self.cache_strategy = mem_cache_strategies[force_mem_cache_release]()
        self.monitor_learning = MonitorLearning()
        self.curves = {}
        self.metric_colors = {}
        if self.training_metrics is not None:
            for tm in self.training_metrics.keys():
                self.curves[f'{tm} (train)'] = []
                if tm not in self.metric_colors:
                    self.metric_colors[tm] = pyplot_colors.pop(0)
        if self.metrics is not None:
            for vm in self.metrics.keys():
                self.curves[f'{vm} (val)'] = []
                if vm not in self.metric_colors:
                    self.metric_colors[vm] = pyplot_colors.pop(0)
        self.model_path = model_path

    def forward(self, x, **kwargs):
        self.cache_strategy()
        prediction = self.model(x, **kwargs)
        if (
            self.validation is not None
            or (
                self.validation is None
                and self.monitor_learning.is_save_time()
                )
        ):
            x['prediction'] = prediction['prediction']
            self.debug_batch(x)
        return prediction

    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data)+1e-10 - np.min(data))

    def debug_batch(self, batch: dict):
        images = {}
        mask = None
        bin_mask_borders = None
        batch_size = batch['prediction'].shape[0]
        for b_i in range(batch_size):
            for k in ['mask', 'prediction', 'image']:
                if k not in batch:
                    continue
                image = batch[k].detach().cpu().numpy()[b_i,0,:,:,:].sum(axis=1)
                image = resize(
                    image,
                    (256,256),
                    preserve_range=True,
                )
                image = self.normalize_data(image)
                # NOTE: 'mask' must be the first element
                if k == 'mask':
                    mask = image
                    bin_mask = (mask > 0.5)
                    bin_mask_borders = (
                        bin_mask.astype(float)
                        - binary_erosion(bin_mask, disk(2)).astype(float)
                    )
                else:
                    assert bin_mask_borders is not None
                    image[bin_mask_borders == 1] = 1
                try:
                    images[b_i] = np.concatenate([images[b_i], image], axis = 1)
                except KeyError:
                    images[b_i] = image
                print(k, torch.unique(batch[k]))
        all_images = np.concatenate(
            [v for _k, v in images.items()],
            axis = 0
        ) # type: np.ndarray
        current_ms = str(int(time.time()*1000))
        if self.validation is not None:
            save_path = self.validation
            current_ms = batch['FileSetId'][0]
        else:
            save_path = join(self.model_path, 'images') # type: ignore

        io.imsave(
            join(save_path, f'{current_ms}.png'),
            (all_images * 255).astype(np.uint8)
        )

    def training_step(self, batch, _batch_idx):
        res = self(batch)
        loss, values = self.loss(batch, res)

        for k in values:
            self.log('Training/'+str(k), values[k].item(), on_step=True,on_epoch=False)

        with torch.no_grad():
            for k in self.training_metrics:
                self.training_metrics[k].update(batch,res)

        return loss

    def on_train_epoch_end(self) -> None:
        metric_results = {
            k:self.training_metrics[k].get()
            for k in self.training_metrics
        }

        if self.training_metrics is not None:
            metric_figures = set()
            for k in self.training_metrics:
                self.log('Training/' + str(k), metric_results[k], on_epoch=True)
                self.training_metrics[k].reset()
                self.curves[k+' (train)'].append(metric_results[k])
                # Save matplotlib plot with all the curves
                metric_figures.add(k)

            # Subplots with one row per metric
            fig, axs = plt.subplots(
                len(metric_figures),
                1,
                figsize=(20, 10*len(metric_figures))
            )
            # Force to be a list
            if not isinstance(axs, list):
                axs = [axs]
            for i, mf in enumerate(metric_figures):
                for k in self.curves:
                    if mf not in k:
                        continue
                    if '(val)' in k:
                        linestyle = '--'
                    else:
                        linestyle = '-'
                    axs[i].plot(
                        self.curves[k],
                        label=k,
                        linestyle=linestyle,
                        color=self.metric_colors[k.split(' ')[0]]
                    )
                axs[i].legend()
                axs[i].set_title(mf)
                axs[i].grid(axis='y')
            fig.savefig(
                join(self.model_path, 'curves.svg'), # type: ignore
                bbox_inches='tight'
            )
            with open(join(self.model_path, 'curves.json'), 'w') as f: # type: ignore
                json.dump(self.curves, f)
            plt.close(fig)

        del metric_results
        self.cache_strategy()

    def validation_step(self, batch, _batch_idx):
        self.cache_strategy()
        with torch.no_grad():
            res = self(batch)

            for k in self.metrics:
                self.metrics[k].update(batch,res)

    def on_validation_epoch_end(self):
        metric_results = {k:self.metrics[k].get() for k in self.metrics}

        for k in self.metrics:
            self.log('Validation/'+str(k), metric_results[k], on_epoch=True)
            self.metrics[k].reset()
            self.curves[k+' (val)'].append(metric_results[k])

        if self.metametrics is not None:
            for k in self.metametrics:
                self.log(str(k), self.metametrics[k].get(metric_results), on_epoch=True)

        del metric_results
        self.cache_strategy()

    def configure_optimizers(self):
        return self.optim

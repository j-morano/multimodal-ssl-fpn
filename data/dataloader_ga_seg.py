from os.path import join
from typing import Dict, List
import json

import numpy as np
from skimage import io

from data.abstract_dataloader import AbstractDataset



class MultimodalGASegDataset(AbstractDataset):
    def __init__(
        self,
        paths: Dict,
        patients: List,
        multiplier=1,
        patches_from_single_image=1,
        transforms=None,
        get_spacing=False,
        reconstruction='None',
    ):
        super().__init__()
        self.path = paths['data']
        self.multiplier = multiplier
        self.patches_from_single_image = patches_from_single_image
        self.transforms = transforms
        self.get_spacing = get_spacing
        self.patients = patients
        self.reconstruction = reconstruction
        with open(paths['info'], 'r') as fp:
            self.visits = json.load(fp)

        self.dataset = self._make_dataset(patients=self.patients)

        self.real_length = len(self.dataset)
        print('# of scans:', self.real_length)

        self.patches_from_current_image = self.patches_from_single_image

    def _load(self, index):
        self.record = self.dataset[index].copy()
        path = self.record['path']
        file_set_id = self.record['FileSetId']

        # Dimensions: front, top, right
        image = np.load(
            join(path, 'bscan_flat.'+file_set_id+'.npy')
        ) # type: np.ndarray
        image = image[None]
        self.record['image'] = image

        if self.get_spacing:
            self.record['spacing'] = np.load(
                join(path, 'spacing.'+file_set_id+'.npy')
            )

        prefix = 'preprocessed_images/bscan_size.'

        if self.reconstruction == 'slo':
            mask = io.imread(
                join(
                    path,
                    prefix+'slo.'+file_set_id+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
        elif self.reconstruction.endswith('faf'):
            # Inverse of FAF, to highlight GA
            mask = io.imread(
                join(
                    path,
                    prefix+'faf.'+file_set_id+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
            if self.reconstruction.startswith('inverted'):
                mask = 1 - mask
        else:
            mask = io.imread(
                join(
                    path,
                    prefix+'mask_faf.'+file_set_id+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
            # Apply threshold
            mask = np.where(mask>=0.5, 1., 0.)
        self.record['mask'] = mask[None,:,None,:]

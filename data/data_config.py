from measure import loss, metrics
from data import dataloader_ga_seg, transforms
from factory_utils import get_factory_adder



add_data_config, data_config_factory = get_factory_adder()


class DefaultConfig:
    h_size = 32
    ch_size = 256
    extra_transforms = []
    reconstruction = 'None'

    paths: dict
    rate_mode: str
    meta_metric_val: dict
    monitor: str

    def get_criterion(self):
        losses = {
            'Dice Loss': loss.DiceLoss(
                output_key='prediction',
                target_key='mask'
            ),
            'BCE loss': loss.BCELoss(
                output_key='prediction',
                target_key='mask'
            ),
        }
        return loss.Mix(losses=losses)


    def get_transforms(self):
        """Get all the transforms using certain crop transforms and
        depending on the type of crop.
        """
        h_size = self.h_size
        ch_size = self.ch_size
        extra_transforms = self.extra_transforms

        to_tensor_transform = transforms.ToTensorDict(transform_keys=['image', 'mask'])

        mirror_transforms = [
            transforms.RandomMirror(transform_keys=['image', 'mask'],dimensions=[1,3])
        ]

        rotation_transforms = [
            transforms.RandomRotation180(keys={ 'image', 'mask' }),
        ]

        crop_transforms = [
            transforms.NewRandomRelCrop(
                reference_key='image',
                transform_keys=['image', 'mask'],
                size=[None, h_size, None, ch_size],
            ),
            transforms.NewRandomRelSize(
                transform_keys=['image', 'mask'],
                fixed_size=[None, h_size, None, ch_size],
            ),
        ]

        intensity_transforms = [
            transforms.AddNoiseAugmentation(transform_keys=['image'],dim=(0,),mu=0.0,sigma=0.2),
            transforms.ContrastAugmentation(transform_keys=['image'],min=0.9, max=1.1),
            transforms.IntensityShift(transform_keys=['image'],min=-0.2, max=0.2),
        ]

        norm_transforms = [
            transforms.ZScoreNormalization(transform_keys=['image'],axis=(2,3)),
        ]

        data_transform = transforms.Compose([
            *norm_transforms,
            *crop_transforms,
            *rotation_transforms,
            *mirror_transforms,
            *intensity_transforms,
            *extra_transforms,
            to_tensor_transform
        ])

        data_transform_val = transforms.Compose([
            *norm_transforms,
            # Use full image
            transforms.NewRandomRelFit(
                transform_keys=['image', 'mask'],
                fit=[None, 16, None, 16]
            ),
            to_tensor_transform
        ])

        return data_transform, data_transform_val


class MMetric:
    def __init__(self, mm):
        self.mm = mm

    def build(self):
        return {
            self.mm: self
        }

    def get(self, m: dict):
        return m[self.mm]


@add_data_config
class Segmentation(DefaultConfig):
    metrics_train = {
        'Dice': metrics.Dice(
            output_key='prediction',
            target_key='mask'
        ),
        'BCE': metrics.BCE(
            output_key='prediction',
            target_key='mask',
            slice=0
        ),
    }

    metrics_val = {
        'Dice': metrics.Dice(
            output_key='prediction',
            target_key='mask'
        ),
        'BCE': metrics.BCE(
            output_key='prediction',
            target_key='mask',
            slice=0
        ),
        'Hausdorff95': metrics.Hausdorff95(
            output_key='prediction',
            target_key='mask',
            slice=0
        ),
    }
    rate_mode = 'minimum'

    monitor = 'Dice'

    paths = {
        'data': '../Multimodal_GA_seg_HRF',
        'split': '../Multimodal_GA_seg_HRF/split_1_be.json',
        'info': '../Multimodal_GA_seg_HRF/hrf_data.json',
    }

    def train_data(self, training_file_list, data_transform, multiplier):
        return dataloader_ga_seg.MultimodalGASegDataset(
            paths=self.paths,
            patients=training_file_list,
            multiplier=multiplier,
            patches_from_single_image=1,
            transforms=data_transform,
            get_spacing=True,
            reconstruction=self.reconstruction,
        )

    def val_data(self, validation_file_list, data_transform_val):
        return dataloader_ga_seg.MultimodalGASegDataset(
            paths=self.paths,
            patients=validation_file_list,
            multiplier=1,
            patches_from_single_image=1,
            transforms=data_transform_val,
            get_spacing=True,
            reconstruction=self.reconstruction,
        )

    meta_metric_val = MMetric('Dice').build()


@add_data_config
class InvertedFAFReconstruction(Segmentation):
    def get_criterion(self):
        losses = {
            'SSIM': loss.SSIMLoss(
                output_key='prediction',
                target_key='mask'
            ),
        }
        return loss.Mix(losses=losses)

    monitor = 'SSIM'

    metrics_train = {
        'SSIM': metrics.SSIM(
            output_key='prediction',
            target_key='mask',
            slice=0
        )
    }

    metrics_val = {
        'SSIM': metrics.SSIM(
            output_key='prediction',
            target_key='mask',
            slice=0
        )
    }

    reconstruction = 'inverted_faf'

    meta_metric_val = MMetric('SSIM').build()


@add_data_config
class SLOReconstruction(InvertedFAFReconstruction):
    reconstruction = 'slo'

import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset


class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-39, 204)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1


class NCLiverVolumeDataset(BaseVolumeDataset):
    # Private non-contrast CT liver-tumor dataset (123 patients, label 2 = tumor).
    # Stats computed from foreground (label>0) of all 123 NC volumes.
    def _set_dataset_stat(self):
        # nii load shape (X, Y, Z); spatial_index=[2,1,0] -> (D, H, W)
        self.intensity_range = (0, 88)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 54.4842
        self.global_std = 14.6129
        self.spatial_index = [2, 1, 0]
        self.do_dummy_2D = True  # 5 mm slice spacing vs 0.68 mm in-plane
        self.target_class = 2


DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "nc_liver": NCLiverVolumeDataset,
}


def create_dataset(
    *,
    data,
    path_prefix,
    data_dir=None,
    split="train",
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop=True,
    do_nnunet_intensity_aug=False,
):
    """Create a dataset object without wrapping in a DataLoader."""
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]

    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )
    return dataset


def load_data_volume(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop=True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
    sampler=None,
):
    dataset = create_dataset(
        data=data,
        path_prefix=path_prefix,
        data_dir=data_dir,
        split=split,
        augmentation=augmentation,
        fold=fold,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if sampler is not None:
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_worker, drop_last=True
        )
    elif deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader

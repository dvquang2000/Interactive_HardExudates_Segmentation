from pathlib import Path

import cv2

import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample,DSample_with_auto_mask

class Retinal_Auto_Mask_Dataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt', auto_masks_dir_name='coarse_mask',
                 **kwargs):
        super(Retinal_Auto_Mask_Dataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self._coarse_mask_path = self.dataset_path / auto_masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        self._coarse_masks_paths = {x.stem: x for x in self._coarse_mask_path.glob('*.*')}

    def get_sample(self, index) -> DSample_with_auto_mask:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])
        coarse_mask_path = str(self._coarse_masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 2].astype(np.int32)
        instances_mask[instances_mask <= 128] = 0
        instances_mask[instances_mask > 128] = 1

        coarse_instances_mask = cv2.imread(coarse_mask_path)[:, :, 2].astype(np.int32)
        coarse_instances_mask = coarse_instances_mask/255
        # coarse_instances_mask[coarse_instances_mask <= 128] = 0
        # coarse_instances_mask[coarse_instances_mask > 128] = 1
        return DSample_with_auto_mask(image, instances_mask, coarse_instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)



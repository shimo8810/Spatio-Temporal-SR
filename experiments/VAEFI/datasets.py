import os
import random
import numpy as np
from PIL import Image

from chainer.dataset import dataset_mixin

def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image / 255.


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)

def _transform(image):
    if random.choice([True, False]):
        image = np.flip(image, 1)

    if random.choice([True, False]):
        image = np.flip(image, 2)

    if random.choice([True, False]):
        image = np.rot90(image, axes=(1, 2))
    return image

class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', dtype=np.float32, is_aug=True):
        self._paths = paths
        self._root = root
        self._dtype = dtype
        self.is_aug = is_aug

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        image = _postprocess_image(image)

        if self.is_aug:
            return _transform(image)
        else:
            return image

class ImageDatasetOnMem(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', dtype=np.float32, is_aug=True):
        self._paths = paths
        self._root = root
        self._dtype = dtype
        self.is_aug = is_aug

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        image = _postprocess_image(image)

        if self.is_aug:
            return _transform(image)
        else:
            return image

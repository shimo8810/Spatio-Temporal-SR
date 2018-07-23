import os

import numpy
from PIL import Image

from chainer.dataset import dataset_mixin

def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image / 255.


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)


class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', dtype=numpy.float32):
        self._paths = paths
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        return _postprocess_image(image)

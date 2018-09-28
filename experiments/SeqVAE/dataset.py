from pathlib import Path
from tqdm import tqdm
from PIL import Image
from chainer.dataset import dataset_mixin
import numpy as np

# PATH 関連
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
DATA_PATH = ROOT_PATH.joinpath('datasets')

class COILDataset(dataset_mixin.DatasetMixin):
    def __init__(self):
        coil_path = DATA_PATH.joinpath('coil-100')
        print(coil_path)
        self.data = []

        print('# load coil-100 data ...')
        for image_path in tqdm(coil_path.glob('*.png')):
            img = np.array(Image.open(image_path)) \
                .astype("f").transpose(2, 0, 1) / 255.0
            self.data.append(img)

        print('# load dataset done!')

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        return self.data[i]

if __name__ == '__main__':
    dataset = COILDataset()
    print(len(dataset))
    img = dataset.get_example(0)
    print(img.shape, img.dtype)

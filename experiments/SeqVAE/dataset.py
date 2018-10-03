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

class SeqCOILDataset(dataset_mixin.DatasetMixin):
    def __init__(self):
        seq_coil_path = DATA_PATH.joinpath('seq-coil-100')
        data = []
        for obj_path in tqdm(seq_coil_path.iterdir()):
            obj_name = obj_path.name
            seqs = []
            for ang in range(0, 360, 5):
                image_path = obj_path.joinpath('{}__{}.png'.format(obj_name, ang))
                image = np.asarray(Image.open(image_path)).astype('f').transpose(2, 0, 1) / 255.0
                seqs.append(image)
            data.append(seqs)
        self.data = np.array(data)
        self.num_obj, self.num_frame, _, _, _ = self.data.shape

    def __len__(self):
        return self.num_obj * self.num_frame

    def get_example(self, i):
        obj = i // self.num_frame
        frame = i % self.num_frame
        frames = [frame, (frame + 1) % self.num_frame, (frame + 2) % self.num_frame]

        return self.data[obj, frames,:]

if __name__ == '__main__':
    dataset = SeqCOILDataset()
    print(len(dataset))
    print(dataset.get_example(71).shape)
    print(dataset.get_example(72).shape)
    print(dataset.get_example(73).shape)
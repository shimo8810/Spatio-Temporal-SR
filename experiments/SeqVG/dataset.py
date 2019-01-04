import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from chainer.dataset import dataset_mixin
import numpy as np
from chainercv import transforms
# PATH 関連
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
DATA_PATH = ROOT_PATH.joinpath('datasets')

class MovingMNISTDataset(dataset_mixin.DatasetMixin):
    """
    Moving-MNISTのデータセット
    データセットディレクトリに存在するmoving_mnist_(train|test).npyファイルは
    元のmnist_test_seq.npyのデータを9:1に分けたデータ

    (64, 64) height, widthの形状で出力される
    """
    def __init__(self, dataset='train'):
        if dataset == 'train':
            mmnist_path = DATA_PATH.joinpath('moving_mnist_train.npy')
        elif dataset == 'test':
            mmnist_path = DATA_PATH.joinpath('moving_mnist_test.npy')
        else:
            raise ValueError('dataset must be "train" or "test".')

        self.data = np.load(mmnist_path) \
                      .reshape(-1, 1, 64, 64).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        return self.data[i]

class SeqMovingMNISTDataset(dataset_mixin.DatasetMixin):
    """
    Moving-MNISTのデータセットのSequenceデータ版
    データセットディレクトリに存在するmoving_mnist_(train|test).npyファイルは
    元のmnist_test_seq.npyのデータを9:1に分けたデータ

    (3, 64, 64) frame, height, widthの形状で出力される
    """
    def __init__(self, dataset='train'):
        if dataset == 'train':
            mmnist_path = DATA_PATH.joinpath('moving_mnist_train.npy')
        elif dataset == 'test':
            mmnist_path = DATA_PATH.joinpath('moving_mnist_test.npy')
        else:
            raise ValueError('dataset must be "train" or "test".')

        self.data = np.load(mmnist_path) \
                      .reshape(-1, 20, 1, 64, 64).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data) * 18

    def get_example(self, i):
        seq_idx = i // 18
        frm_idx = i %  18
        return self.data[seq_idx, frm_idx:frm_idx+3]

class SeqCOILDataset2(dataset_mixin.DatasetMixin):
    def __init__(self, dataset='train'):
        print("loading {}".format(dataset))

        if dataset == 'train':
            coil_path = DATA_PATH.joinpath('sequence_coil_100_train.npy')
        elif dataset == 'test':
            coil_path = DATA_PATH.joinpath('sequence_coil_100_test.npy')
        else:
            raise ValueError("'dataset' must be 'train' or 'test'.")

        self.data = np.load(coil_path) \
                      .transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.0

        self.num_obj, self.num_fr, _, _, _ = self.data.shape

    def __len__(self):
        return self.num_obj * self.num_fr

    def get_example(self, i):
        obj = i // self.num_fr
        fr = i % self.num_fr
        frs = [fr, (fr + 1) % self.num_fr, (fr + 2) % self.num_fr]
        seq = self.data[obj, frs]
        return self.argument(seq)

    def argument(self, seq):
        # pca lightning
        eigen_value = np.array((0.2175, 0.0188, 0.0045))
        eigen_vector = np.array((
            (-0.5675, -0.5808, -0.5836),
            (0.7192, -0.0045, -0.6948),
            (0.4009, -0.814,  0.4203)))
        alpha = np.random.normal(0, 0.25, size=3)
        seq = seq.copy() + eigen_vector.dot(eigen_value * alpha) \
                            .reshape((1, -1, 1, 1)).astype(np.float32)

        # flip h
        if random.choice([True, False]):
            seq = np.flip(seq, 2)

        # flip v
        if random.choice([True, False]):
            seq = np.flip(seq, 3)

        # rot 90
        if random.choice([True, False]):
            seq = np.rot90(seq, axes=(2, 3))

        return seq

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
    def __init__(self, data_aug=False):
        self.data_aug = data_aug
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

        # 拡張データ
        if data_aug:
            aug_coil_path = DATA_PATH.joinpath('seq-coil-scale-100')
            for obj_path in tqdm(aug_coil_path.iterdir()):
                obj_name = obj_path.name
                seqs = []
                for i in range(72):
                    image_path = obj_path.joinpath('{}__{}.png'.format(obj_name, i))
                    image = np.asarray(Image.open(image_path)).astype('f').transpose(2, 0, 1) / 255.0
                    seqs.append(image)
                data.append(seqs)

            aug_coil_path = DATA_PATH.joinpath('seq-coil-scale-100')
            for obj_path in tqdm(aug_coil_path.iterdir()):
                obj_name = obj_path.name
                seqs = []
                for i in range(72):
                    image_path = obj_path.joinpath('{}__{}.png'.format(obj_name, i))
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

        seq = self.data[obj, frames,:]
        if self.data_aug:
            return self.data_transform(seq)
        return seq

    def data_transform(self, seq):
        # flip h
        if random.choice([True, False]):
            seq = np.flip(seq, 2)

        # flip v
        if random.choice([True, False]):
            seq = np.flip(seq, 3)

        # rot 90
        if random.choice([True, False]):
            seq = np.rot90(seq, axes=(2, 3))

        return seq


if __name__ == '__main__':
    data = MovingMNISTDataset()

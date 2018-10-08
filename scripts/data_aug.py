from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# PATH 関連
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent
DATA_PATH = ROOT_PATH.joinpath('datasets')

def main():
    # データ読み込み
    seq_coil_path = DATA_PATH.joinpath('seq-coil-100')
    for obj_path in tqdm(seq_coil_path.iterdir()):
        obj_name = obj_path.name
        image = Image.open(obj_path.joinpath('{}__{}.png'.format(obj_name, 0)))

        # 拡縮による拡張
        # 保存用パス
        scale_path = DATA_PATH.joinpath('seq-coil-scale-100/{}'.format(obj_name))
        scale_path.mkdir(parents=True, exist_ok=True)

        for i, x in enumerate(np.linspace(-64, 64, 72)):
            aff = (1, 0, x, 0, 1, 0)
            img_tf = image.transform(image.size, Image.AFFINE, aff, Image.BICUBIC)
            img_tf.save(scale_path.joinpath('{}__{}.png'.format(obj_name, i)))

        # 平行移動による拡張
        trans_path = DATA_PATH.joinpath('seq-coil-trans-100/{}'.format(obj_name))
        trans_path.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(np.linspace(0.4, 2.0, 72)):
            m1, m2, _ = np.linalg.inv(
                np.array([[r, 0, 64-64*r], [0, r, 64-64*r],[0,0,1]]))
            aff = np.hstack((m1,m2))
            img_tf = image.transform(image.size, Image.AFFINE, aff, Image.BICUBIC)
            img_tf.save(trans_path.joinpath('{}__{}.png'.format(obj_name, i)))

if __name__ == '__main__':
    main()

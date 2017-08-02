import random
from sys import argv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage import io, transform


def gen_npz(paths, target_path):
    target_path = Path(target_path)
    print(target_path)

    xs = np.zeros((len(paths), 224, 224, 3), dtype=np.float32)
    ys = np.zeros((len(paths), 2), dtype=np.uint8)
    for i, path in enumerate(tqdm(paths)):
        img = io.imread(path)
        y = 0 if path.stem[:4] == 'cats' else 1
        xs[i] = transform.resize(img, (224, 224), mode='edge')
        ys[i][y] = 1

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(target_path), xs=xs, ys=ys)


def main():
    dataset = Path(argv[1]).expanduser().absolute()
    imgs = list(dataset.glob('*.jpg'))

    random.shuffle(imgs)
    pivot = len(imgs) * 4 // 5
    train, val = imgs[:pivot], imgs[pivot:]

    gen_npz(train, 'train.npz')
    gen_npz(val, 'val.npz')


if __name__ == '__main__':
    main()

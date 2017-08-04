import random
import argparse
from sys import argv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage import io, transform

parser = argparse.ArgumentParser(description='Cats vs Dogs')
parser.add_argument(
    'source', metavar='SOURCE_DIR', help='path to train dir of dataset')
parser.add_argument(
    '--num', '-n', type=int, default=10000, help='number of samples to use')
parser.add_argument(
    '--target',
    '-t',
    metavar='TARGET_DIR',
    default='./',
    help='path to target dir')
args = parser.parse_args()


def gen_npz(paths, target_path):
    print(target_path)

    xs = np.zeros((len(paths), 224, 224, 3), dtype=np.float32)
    ys = np.zeros((len(paths)), dtype=np.uint8)
    for i, path in enumerate(tqdm(paths)):
        img = io.imread(path)
        xs[i] = transform.resize(img, (224, 224), mode='edge')
        ys[i] = 0 if 'cat' in path.name else 1

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(target_path), xs=xs, ys=ys)


def main():
    dataset = Path(args.source).expanduser().absolute()
    imgs = random.sample(list(dataset.glob('*.jpg')), args.num)

    random.shuffle(imgs)
    pivot = len(imgs) * 4 // 5
    train, val = imgs[:pivot], imgs[pivot:]

    target_dir = Path(args.target)
    gen_npz(train, target_dir / 'train.npz')
    gen_npz(val, target_dir / 'val.npz')


if __name__ == '__main__':
    main()

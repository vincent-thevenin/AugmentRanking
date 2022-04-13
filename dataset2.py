import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import msgpack
from tqdm import tqdm
import pyarrow as pa

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        # env = self.env
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        
        # # load image
        # imgbuf = unpacked[0]
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')
        img = unpacked[0]

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pickle(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def folder2lmdb(dpath, lmdb_path, name="train", write_frequency=5000, num_workers=16):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(
        directory,
        # loader=raw_reader,
        transform=transforms.Compose((
            transforms.Resize(224),
            transforms.CenterCrop(224),
        )),
    )
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    # lmdb_path = osp.join(lmdb_path, "%s.lmdb" % name)
    # isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(data_loader)):
        # print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pickle((image, label)))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, lmdb_path=args.out, num_workers=args.procs, name=args.split)

from tkinter.tix import Tree
import lmdb
import os
import pickle
from PIL import ImageFile
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

ds_train = ImageFolder(
    os.getcwd() + '/data/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train',
    transform=transforms.Compose((
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )),
)
loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True 
)

#init lmdb
lmdb_path = os.getcwd() + '/data/imagenet_train_lmdb'
db = lmdb.open(
    lmdb_path,
    map_size=1099511627776 * 2,
    meminit=False,
)

txn = db.begin(write=True)
for idx, (x, y) in enumerate(tqdm(loader)):
    y = nn.functional.one_hot(y, 1000).float()
    txn.put(str(idx).encode('ascii'), pickle.dumps((x,y)))

    if idx % 5000 == 0:
        txn.commit()
        txn = db.begin(write=True)
    
    if idx == 5000:
        break
    

txn.commit()
keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
with db.begin(write=True) as txn:
    txn.put(b'__keys__', pickle.dumps(keys))
    txn.put(b'__len__', pickle.dumps(len(keys)))

print("Flushing database ...")
db.sync()
db.close()
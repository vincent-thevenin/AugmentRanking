# from ctypes.wintypes import tagMSG
from tabnanny import verbose
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler, RMSprop
import torch.nn as nn
from tqdm import tqdm

from dataset2 import ImageFolderLMDB
from model import Net


dataset_name = "ImageNet"
epochs = 100
batch = 2**6
lr = 0.256 / 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentationH = transforms.RandomHorizontalFlip(p=0.5)
augmentationV = transforms.RandomVerticalFlip(p=0.5)

if dataset_name == "CIFAR10":
    ds_train = CIFAR10(
        os.getcwd() + '/data',
        train=True,
        download=True,
        transform=transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            augmentationH
        ))
    )
    ds_val = CIFAR10(
        os.getcwd() + '/data',
        train=False,
        download=True,
        transform=transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ))
    )
elif dataset_name == "ImageNet":
    ds_train = ImageFolderLMDB(
        os.getcwd() + '/data/imagenet_lmdb_train',
        transform=transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            augmentationH,
        )),
    )

    ds_val, ds_train = torch.utils.data.random_split(ds_train, [25000, len(ds_train) - 25000], generator=torch.Generator().manual_seed(42))

loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    pin_memory=True 
)
val_loader = torch.utils.data.DataLoader(
    ds_val,
    batch_size=batch,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
total_iters = len(loader)

model = EfficientNet.from_name('efficientnet-b0')#.to(device)
# model = nn.Sequential(
#     nn.Linear(1000, 10),
#     nn.LeakyReLU(0.1)
# ).to(device)
# model = Net().to(device)

optimizer = RMSprop(list(model.parameters()), lr=lr, weight_decay=0.9, momentum=0.9)
scheduler_begin = lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

lr_update_interval = int(2.4 * total_iters)
loss_fn = nn.CrossEntropyLoss()

scaler = GradScaler()


iterations = 0
for e in range(epochs):
    pbar = tqdm(loader)
    # model.train()
    for x, y in pbar:
        optimizer.zero_grad()
        with autocast():
            x = x.to(device)
            y = y.to(device)
            if dataset_name == "ImageNet":
                y = nn.functional.one_hot(y, 1000).float()
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        iterations += 1
        if iterations % lr_update_interval == 0 and e >= 5:
            scheduler.step()
        pbar.set_postfix(
            Loss=loss.item()
        )
    
    if e < 5:
        scheduler_begin.step()

    # model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        print(f'Epoch {e+1}/{epochs}')
        print(f'Accuracy: {correct/total:.4f}')
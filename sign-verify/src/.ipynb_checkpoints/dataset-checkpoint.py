import torch as T
from torch.nn.modules import transformer
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os

import config


def read_file(filename):
    with open(filename) as fp:
        return fp.readlines()


def read_dataset(filename):
    lines = read_file(filename)
    rows = []
    for line in lines:
        img1, img2, label = line.strip().split(',')
        rows.append((img1, img2, label))
    return rows


def load_image(filename, transformer):
    img1 = Image.open(filename)
    return transformer(img1)



class SignDataset(data.Dataset):
    def __init__(self, filename, basedir):
        self.filename = filename
        self.basedir = basedir
        self.rows = read_dataset(filename)
        self.transformer = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor()
        ])
        self.images = {}
        for row in self.rows:
            img1 = os.path.join(self.basedir, row[0])
            img2 = os.path.join(self.basedir, row[1])
            if not img1 in self.images:
                self.images[img1] = load_image(img1, self.transformer)
            if not img2 in self.images:
                self.images[img2] = load_image(img2, self.transformer)

    def __getitem__(self, i):
        img1 = os.path.join(self.basedir, self.rows[i][0])
        img2 = os.path.join(self.basedir, self.rows[i][1])

        return (T.cat([self.images[img1], self.images[img2]]).float().to(config.device), 
            T.tensor(int(self.rows[i][2])).long().to(config.device))

    def __len__(self):
        return len(self.rows)


import torch as T
from torch.nn.modules import transformer
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from glob import glob
from collections import defaultdict
import itertools
from tqdm import tqdm

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


def readLines(filename):
    with open(filename) as fp:
        return fp.readlines()


def generate_image(coords, transformer):
    max_x = max([x for x, y in coords])
    max_y = max([y for x, y in coords])

    # print(max_x+10,max_y+10)

    image = Image.new('L', (max_x+10, max_y+10), 255)
    imageDraw = ImageDraw.Draw(image)
    for i in range(1, len(coords)):
        px = coords[i-1]
        p = coords[i]
        imageDraw.line((px, p), fill=0, width=2)
    image = transformer(image)
    return image


def buildFingerTensor(filename, transformer):
    coords = [(int(x), int(y)) for x, y in [line.strip().split()[:2] for line in readLines(filename)[1:]]]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x = min(xs) - 10
    min_y = min(ys) - 10
    final_coords = [(x-min_x, y-min_y) for x, y in coords]
    return generate_image(final_coords, transformer)


class DeepSignDBFinger:
    def __init__(self, basedir, buildFingerTensorFn = buildFingerTensor):
        self.files = list(glob(f'{basedir}/*.txt'))
        self.userToSign = defaultdict(list)
        self.userToFsign = defaultdict(list)
        for file in self.files:
            parts = os.path.basename(file).split('_')
            user = parts[2]
            signType = parts[6]
            if signType == 'sign':
                self.userToSign[user].append(file)
            else:
                self.userToFsign[user].append(file)
        self.users = list(self.userToSign.keys())
        self.tensors = {}
        self.transformer = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor()
        ])
        self.buildFingerTensor = buildFingerTensorFn
        self.buildTensors()
    
    def buildTensors(self):
        for file in tqdm(self.files, unit='file'):
            self.tensors[file] = self.buildFingerTensor(file, self.transformer)

    def getUserSign(self, user):
        return self.userToSign[user]
    
    def getUserFsign(self, user):
        return self.userToFsign[user]
    
    def getUsers(self):
        return self.users
    
    def getTensor(self, file):
        return self.tensors[file]


class DeepSignDBStylus:
    def __init__(self, basedir):
        pass
    
    def getUserSign(self, user):
        pass
    
    def getUserFsign(self, user):
        pass
    
    def getUsers(self):
        pass

    def getTensor(self, file):
        pass


class DeepSignDBDataset(data.Dataset):
    def __init__(self, db):
        self.db = db
        self.validUsers = [user for user in self.db.getUsers() if len(self.db.getUserSign(user)) != 0 or len(self.db.getUserFsign(user)) != 0]
        self.length = 0
        self.pairs = []
        for user in self.validUsers:
            userSign = self.db.getUserSign(user)
            userFsign = self.db.getUserFsign(user)
            validPairs = [(a, b, 1) for a, b in list(itertools.combinations(userSign, 2))]
            invalidPairs = [(a, b, 0) for a in userSign for b in userFsign]
            self.pairs.extend(validPairs)
            self.pairs.extend(invalidPairs)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        sign, v_sign, label = self.pairs[index]
        sign_tensor = self.db.getTensor(sign)
        v_sign_tensor = self.db.getTensor(v_sign)
        # print(repr(sign), repr(v_sign), sign_tensor, v_sign_tensor)
        return (T.cat([sign_tensor, v_sign_tensor]).float().to(config.device), 
            T.tensor(label, dtype=T.long).to(config.device))







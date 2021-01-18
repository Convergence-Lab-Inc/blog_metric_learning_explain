import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sets = ["chihuahua", "collie", "dalmatian","papillon dog", "sheltie"]

class DogDataset(Dataset):
    def __init__(self, transform=None, train=False):
        self.image_dict = defaultdict(list)
        self.records = []
        path = "dataset/"
        if train:
            path += "train/"
        else:
            path += "test/"
        for item in sets:
            images = glob(path + item + "/*")
            self.image_dict[item] = images
            for img in images:
                self.records += [[item, img]]
        self.transform = transform
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        label, img = self.records[index]
        img = Image.open(img)
        pos = np.random.choice(self.image_dict[label])
        pos = Image.open(pos)
        while True:
            neg_key = np.random.choice(sets)
            if neg_key != label:
                break
        neg = np.random.choice(self.image_dict[neg_key])
        neg = Image.open(neg)
        if self.transform:
            img = self.transform(img)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return img, pos, neg, label

def test_collator(batch):
    batch_x = []
    batch_label = []
    batch_img = []
    for item in batch:
        label, x, img = item
        batch_label = [label]
        batch_x = [x.unsqueeze(0)]
        batch_img = [img]
    batch_x = torch.cat(batch_x, axis=0)
    return batch_label, batch_x, batch_img

class DogTestDataset(Dataset):
    def __init__(self, transform=None, train=False):
        self.image_dict = defaultdict(list)
        self.records = []
        path = "dataset/"
        if train:
            path += "train/"
        else:
            path += "test/"
        for item in sets:
            images = glob(path + item + "/*")
            self.image_dict[item] = images
            for img in images:
                self.records += [[item, img]]
        self.transform = transform
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        label, img = self.records[index]
        x = Image.open(img)
        if self.transform:
            x = self.transform(x)

        return label, x, img

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.RandomAffine(15, translate=(0.2, 0.2), scale=(0.55, 1.5), shear=15),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
        ])
    dataset = DogDataset(transform)
    img, pos, neg, label = dataset[0]
    print(label, img.shape, pos.shape, neg.shape)
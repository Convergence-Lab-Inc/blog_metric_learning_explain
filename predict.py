from model import DogModel
from data import DogTestDataset, test_collator

import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


def change_coordinate(x):
    x =  (x + 16) * 15
    x = x.astype(np.int32)
    # x = (x[0], x[0]+32, x[1]+32, x[1])
    return (x[0], x[1])

def predict():
    epochs = 20
    batch_size = 1

    test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    test_dataset = DogTestDataset(transform=test_transform, train=False)
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = DogModel(metric_size=2)
    checkpoint = torch.load("net.pt")
    net.load_state_dict(checkpoint["net"])
    net.to(device)

    im = Image.new(mode="RGBA", size=(512, 512), color=(255, 255, 255, 255))

    net.eval()
    test_loss = 0
    for batch in tqdm(testloader):
        label, x, img = batch
        with torch.no_grad():
            x = x.to(device)
            x = net(x)
        x = x.cpu().numpy()
        im2 = Image.open(img[0])
        im2 = im2.resize((64, 64)).convert("RGBA")
        coord = change_coordinate(x[0])
        Image.Image.paste(im, im2, coord)
    im.save("img.png")

if __name__ == "__main__":
    predict()
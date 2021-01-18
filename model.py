import torch
from torch import nn

class DogModel(nn.Module):
    def __init__(self, metric_size=2):
        super(DogModel, self).__init__()
        self.basenet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.basenet.fc = nn.Linear(512, metric_size)

    def __call__(self, x):
        x = self.basenet(x)
        return x
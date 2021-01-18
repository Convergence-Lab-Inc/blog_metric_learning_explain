from model import DogModel
from data import DogDataset
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

def distance(x1, x2):
    return 1.0 - F.cosine_similarity(x1, x2, dim=1)

def check_grads(params):
    grad = 0
    i = 0
    for p in params:
        if p.grad is not None:
            grad += p.grad.norm()
            i += 1
    return grad / i

def train():
    epochs = 20
    batch_size = 32
    train_transform = transforms.Compose([
            # transforms.RandomAffine(15, translate=(0.2, 0.2), scale=(0.5, 1.5), shear=15),
            transforms.RandomResizedCrop((128, 128)),
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    train_dataset = DogDataset(transform=train_transform, train=True)
    test_dataset = DogDataset(transform=test_transform, train=False)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = DogModel(metric_size=2)
    net.to(device)

    criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance, margin=0.5)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    for ep in range(epochs):
        train_loss = 0
        net.train()
        for batch in tqdm(trainloader):
            x, pos, neg, label = batch
            x = x.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            x = net(x)
            pos = net(pos)
            neg = net(neg)
            loss = criterion(x, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {ep*1}: Train loss: {train_loss / len(trainloader)}")
        net.eval()
        test_loss = 0
        for batch in tqdm(testloader):
            x, pos, neg, label = batch
            with torch.no_grad():
                x = x.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                x = net(x)
                pos = net(pos)
                neg = net(neg)
                loss = criterion(x, pos, neg)
            test_loss += loss.item()
        print(f"Epoch {ep*1}: Test loss: {test_loss / len(testloader)}")
        torch.save({
            "net": net.state_dict(),
            "optimzer": optimizer.state_dict()
        }, "net.pt")

if __name__ == "__main__":
    train()
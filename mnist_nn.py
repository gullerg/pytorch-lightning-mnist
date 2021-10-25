import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)

class ResNet(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h1 + h2)
        logits = self.l3(do)
        return logits
model = ResNet(0.1)



optimiser = optim.SGD(model.parameters(), lr=1e-2)

loss = nn.CrossEntropyLoss()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        l = model(x)
        J = loss(l, y)

        model.zero_grad()

        J.backward()

        optimiser.step()

        losses.append(J.item())
    print(f'Epoch: {epoch +1 }, train loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in train_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        with torch.no_grad():
            l = model(x)
        J = loss(l, y)
        losses.append(J.item())

    print(f'Epoch: {epoch + 1}, val loss: {torch.tensor(losses).mean():.2f}')

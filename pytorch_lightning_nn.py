import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torchmetrics

accuracy = torchmetrics.Accuracy()

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h1 + h2)
        logits = self.l3(do)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        logits = self(x)
        J = self.loss(logits, y)

        acc = accuracy(logits, y)
        pbar = {"train_acc": acc}

        return {'loss': J, "progress_bar": pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results["progress_bar"]["val_acc"] = results["progress_bar"]["train_acc"]
        del results["progress_bar"]["train_acc"]
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x["progress_bar"]["val_acc"] for x in val_step_outputs]).mean()

        pbar = {"avg_val_acc": avg_val_acc}

        return {'val_loss': avg_val_loss, "progress_bar": pbar}

    def prepare_data(self):
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        dataset = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        self.train_dataset, self.val_dataset = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=32)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=32)
        return val_loader

model = ResNet()

trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5)
trainer.fit(model)
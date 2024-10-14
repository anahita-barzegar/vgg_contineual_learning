import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from pixelsnail import PixelSNAIL


def train_pixelcnn(epoch, loader, model, optimizer, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out = model(img)
        loss = criterion(out, img)
        loss.backward()

        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == img).float()
        accuracy = correct.sum() / img.numel()

        loader.set_description(
            (f'epoch: {epoch + 1}; loss: {loss.item():.5f}; ' f'acc: {accuracy:.5f}')
        )
    return model


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()




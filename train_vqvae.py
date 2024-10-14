from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from scheduler import CycleScheduler
from torch import nn, optim
from vqvae import VQVAE
import distributed as dist
from tqdm import tqdm
import general_utils
import torchvision
import argparse
import torch
import sys
import os


def train_generator(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    performance_result = []
    latent_values = []
    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        # latent_values.append([model.encode(img)[-1], label])
        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            performance_result.append(
                {'epoch': epoch + 1, 'mse': recon_loss.item(),
                 'latent': latent_loss.item(), 'avg_mse': mse_sum / mse_n,
                 'lr': lr
                 })
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]
                sample_label = label[:sample_size]
                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"data/sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1)
                )
                torch.cuda.empty_cache()
                model.train()

    return model, latent_values

    general_utils.plot_results(performance_result, 'epoch', 'avg_mse', 'generator')

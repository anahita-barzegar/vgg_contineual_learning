import torch
from torch import nn
from torch.nn import functional as F
import distributed as dist_fn


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True))
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, in_channel)

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel * 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel * 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, class_id):
        class_embed = self.embedding(class_id).unsqueeze(2).unsqueeze(3)
        class_embed = class_embed.expand(-1, -1, input.size(2), input.size(3))
        input = torch.cat([input, class_embed], 1)
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, out_channel)

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, class_id):
        out = self.blocks(input)
        class_embed = self.embedding(class_id).unsqueeze(2).unsqueeze(3)
        out = out + class_embed.expand(-1, -1, out.size(2), out.size(3))
        return out


class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512,
                 decay=0.99, n_classes=10):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4, n_classes=n_classes)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2, n_classes=n_classes)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2, n_classes=n_classes)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4,
                           n_classes=n_classes)

    def forward(self, input, class_id):
        quant_t, quant_b, diff, _, _, _ = self.encode(input, class_id)
        dec = self.decode(quant_t, quant_b, class_id)
        return dec, diff

    def encode(self, input, class_id):
        enc_b1 = self.enc_b(input, class_id)
        enc_t = self.enc_t(enc_b1, class_id)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t, class_id)
        enc_b = torch.cat([dec_t, enc_b1], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b, enc_b1, enc_b

    def decoder_to_quant(self, enc_b, class_id):
        enc_t = self.enc_t(enc_b, class_id)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t, class_id)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b

    def decode(self, quant_t, quant_b, class_id):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant, class_id)
        return dec

    def generate_sample_for_class(self, class_id, image_size=(32, 32)):
        device = next(self.parameters()).device
        random_latent_t = torch.randn(1, self.quantize_t.dim, image_size[0] // 4, image_size[1] // 4).to(device)
        random_latent_b = torch.randn(1, self.quantize_b.dim, image_size[0] // 2, image_size[1] // 2).to(device)
        sample = self.decode(random_latent_t, random_latent_b, torch.tensor([class_id]).to(device))
        return sample

    def generate_random_samples(self, num_samples, image_size=(32, 32)):
        device = next(self.parameters()).device
        samples = []
        for _ in range(num_samples):
            class_id = torch.randint(0, self.dec.embedding.num_embeddings, (1,)).to(device)
            random_latent_t = torch.randn(1, self.quantize_t.dim, image_size[0] // 4, image_size[1] // 4).to(device)
            random_latent_b = torch.randn(1, self.quantize_b.dim, image_size[0] // 2, image_size[1] // 2).to(device)
            sample = self.decode(random_latent_t, random_latent_b, class_id)
            samples.append(sample)
        return torch.cat(samples)


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.utils as vutils
import os


# Define your training function
def train_vqvae(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, diff = model(inputs, labels)
            recon_loss = criterion(outputs, inputs)
            loss = recon_loss + diff.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, diff = model(inputs, labels)
                recon_loss = criterion(outputs, inputs)
                loss = recon_loss + diff.mean()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'vqvae_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")


# CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Initialize the model, loss function, and optimizer
model = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99,
              n_classes=10)

# Train the model
train_vqvae(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device='cuda')


'''
#instances[n_samples] + instances[n_samples + 1] - instances[n_samples + 2]
import matplotlib.pyplot as plt

numpy_image = instances[2].numpy()
numpy_image = np.transpose(numpy_image, (1, 2, 0))
plt.imshow(numpy_image)

# Plot the image
plt.axis('off')  # Turn off axis labels
plt.savefig(f'data/imaginations/0_2_sample.png')
'''

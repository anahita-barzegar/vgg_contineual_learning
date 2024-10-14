import torch
import torchvision
import torchvision.transforms as transforms

# Step 1: Import Required Libraries
# Ensure you have `torch` and `torchvision` installed. You can install them via pip:
# pip install torch torchvision

# Step 2: Set Up Transformations
# Define the transformations for different datasets
transform_dict = {
    'MNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std for MNIST
    ]),
    'CIFAR10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for CIFAR-10
    ])
}


# Step 3: Create Function to Load Dataset
def load_dataset(dataset_name, batch_size=64, num_workers=2):
    if dataset_name not in transform_dict:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transform_dict[dataset_name]

    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, test_dataset, train_dataset






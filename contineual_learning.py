import argparse
import sys
import os
import time
from torch.utils.data import DataLoader, Subset
import random
import torch
import variable as variable
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from vgg import VGG, VGGOriginal, VGG16Classifier
from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
import torchvision
import general_utils
from train_vqvae import train_generator
from train_pixelCNN import train_pixelcnn
from pixelsnail import PixelSNAIL

from train_classifier import train_classifier
from classifier import Classifier
from svm_classifier import svmClassifier
from data_loader import load_dataset

torch.cuda.empty_cache()


# import gc
#
# del variable  # Delete unnecessary variable
# gc.collect()  # Force garbage collectioncc
#
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def split_data(trainset, class_number_size):
    # Define the number of classes in CIFAR-10
    num_classes = 10
    if class_number_size == num_classes:
        return [trainset]
    # Create subsets for every pair of two classes
    class_subsets = []
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    j = 0
    while j < (num_classes):
        class_indices = [i for i, (_, label) in enumerate(trainset) if label in [j, j + 1]]
        class_subset = Subset(trainset, class_indices)
        print("class_subset: ", [j, j + 1])
        class_subsets.append(class_subset)
        j = j + class_number_size

    # Example usage
    print("Number of pairs of class subsets:", len(class_subsets))

    return class_subsets


def get_subset(loader, num_samples):
    indices = list(range(len(loader.dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    subset = Subset(loader.dataset, subset_indices)
    subset.dataset = subset.dataset[:num_samples]
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=True)


# If you want to create subsets of the datasets directly
def get_dataset_subset(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    return Subset(dataset, subset_indices)


def train_vqvae(args):
    # device = "cuda"
    device = "cuda"

    args.distributed = dist.get_world_size() > 1
    dataset_name = args.dataset  # Change this to 'CIFAR10' to load CIFAR-10

    if not args.continual_learning:
        # Step 4: Use the Function to Load Specific Dataset
        loader, test_loader, test_set, train_set = load_dataset(dataset_name)
        data_subsets = split_data(train_set, 10)

    # # Create subsets
    # loader = get_subset(loader, 5000)
    # test_loader = get_subset(test_loader, 1000)
    #
    # train_set = get_dataset_subset(train_set, 5000)
    # test_set = get_dataset_subset(test_set, 1000)

    elif args.continual_learning:
        # Load cfar-10 dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        data_subsets = split_data(train_set, 2)

        loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    # loader = DataLoader(
    #     dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    # )

    # data_subsets = split_data(train_set)
    num_classes = 10

    if args.imagination_level == 'data':
        channel = 3
        # c_model = VGG16Classifier(num_classes=10, channel=channel).to(device)
        c_model = VGGOriginal(num_classes=num_classes, channel=channel).to(device)

    elif args.imagination_level == 'enc_b':
        channel = 192
        c_model = VGG(num_classes=num_classes, channel=channel).to(device)

    elif args.imagination_level == 'enc_b1':
        channel = 128
        c_model = VGG(num_classes=num_classes, channel=channel).to(device)

    if args.model_name == 'cnn':
        c_model = Classifier(num_classes=num_classes, channel=channel).to(device)

    g_model = VQVAE().to(device)
    prev_c_model = None
    prev_g_model = None
    prev_p_model = None
    # Task Incremental Learning
    for task_number, data in enumerate(data_subsets):
        if args.continual_learning and task_number > 2:
            break

        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

        if prev_g_model is not None:
            g_model = prev_g_model

        if prev_c_model is not None:
            c_model = prev_c_model
            # general_utils.plot_model(data[0], model, 'generator')

        # if prev_p_model is not None:
        #     p_model = prev_p_model
        # else:
        #     p_model = PixelSNAIL([32, 32], 256, 128, 5, 2, 4, 128)

        general_utils.print_model_info(g_model)

        # if args.distributed:
        #     g_model = nn.parallel.DistributedDataParallel(
        #         g_model,
        #         device_ids=[dist.get_local_rank()],
        #         output_device=dist.get_local_rank(),
        #     )
        #
        # g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr)
        # scheduler = None
        # if args.sched == "cycle":
        #     scheduler = CycleScheduler(
        #         g_optimizer,
        #         args.lr,
        #         n_iter=len(loader) * args.epoch,
        #         momentum=None,
        #         warmup_proportion=0.05,
        #     )

        # p_model = p_model.to(device)
        #
        # p_optimizer = optim.Adam(p_model.parameters(), lr=1e-3)

        #
        # for i in range(10):
        #     p_model = train_pixelcnn(i, loader, p_model, p_optimizer, device)
        #     torch.save(p_model.state_dict(), f'checkpoint/cfar10_pixelcnn_{str(i + 1).zfill(3)}.pt')

        # for i in range(args.epoch):
        #     g_model, latent_values = train_generator(i, loader, g_model, g_optimizer, scheduler, device)
        #     # g_model.sample(2, 1)
        #     if dist.is_primary():
        #         torch.save(g_model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")
        epoch_number = args.classifier_epoch
        sleep_param = args.sleep
        g_model = VQVAE().to(device)
        g_model.load_state_dict(
            torch.load("/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vqvae_200.pt"))
        g_model.eval()
        test_latent_values = []
        train_latent_values = []
        if args.imagination_level not in ['quant', 'data']:
            iml = {'enc_b': -1, 'enc_b1': -2, 'quant_b': 0, 'quant_t': 1}
            for i, (img, label) in enumerate(test_loader):
                with torch.no_grad():
                    test_latent_values.append(
                        [g_model.encode(img.to(device))[iml[args.imagination_level]], label.to(device)])

            for i, batch in enumerate(loader):
                img, label = batch
                with torch.no_grad():
                    train_latent_values.append(
                        [g_model.encode(img.to(device))[iml[args.imagination_level]], label.to(device)])

            c_model = train_classifier(c_model, train_latent_values, test_latent_values, num_classes, channel,
                                       args.imagination_level, args.imagination_param,
                                       task_number, epoch_number, args.model_name, dataset=dataset_name,
                                       sleep_param=sleep_param, replay_iter_param=args.replay_iter,
                                       contineual_learning=args.continual_learning,
                                       replay_batch_number=args.replay_batch_number,
                                       replay_batch_size=args.replay_batch_size)

        elif args.imagination_level == 'data':
            c_model = train_classifier(c_model, loader, test_loader, num_classes, channel,
                                       args.imagination_level, args.imagination_param,
                                       task_number, epoch_number, args.model_name, dataset=dataset_name,
                                       sleep_param=sleep_param, replay_iter_param=args.replay_iter,
                                       contineual_learning=args.continual_learning,
                                       replay_batch_number=args.replay_batch_number,
                                       replay_batch_size=args.replay_batch_size)

        else:
            for i, (img, label) in enumerate(test_loader):
                test_latent_values.append([[g_model.encode(img)[0], g_model.encode(img)[1]], label])

            for i, (img, label) in enumerate(loader):
                train_latent_values.append([[g_model.encode(img)[0], g_model.encode(img)[1]], label])

            c_model = train_classifier(c_model, train_latent_values, test_latent_values, num_classes, channel,
                                       args.imagination_level, args.imagination_param,
                                       task_number, epoch_number, args.model_name, dataset=dataset_name,
                                       sleep_param=sleep_param, replay_iter_param=args.replay_iter,
                                       contineual_learning=args.continual_learning,
                                       replay_batch_number=args.replay_batch_number,
                                       replay_batch_size=args.replay_batch_size)

        # c_model = svmClassifier(latent_values[0])
        prev_g_model = g_model
        prev_c_model = c_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=0)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--classifier_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--channel", type=str, default=192)
    parser.add_argument("--dataset", type=str, default='CIFAR10')  # MNIST,CIFAR10
    parser.add_argument("--sleep", type=str, default=False)  # True,False
    parser.add_argument("--replay_iter", type=int, default=20)
    parser.add_argument("--replay_batch_number", type=int, default=10)
    parser.add_argument("--replay_batch_size", type=int, default=100)
    parser.add_argument("--model_name", type=str, default='vgg')  # vgg,cnn
    parser.add_argument("--imagination_level", type=str, default="enc_b")  # data,enc_b,quant
    parser.add_argument("--imagination_param", type=str, default="None")  # None,arithmetic,marcov,ari+marc
    parser.add_argument("--continual_learning", type=str, default=True)  # True,False

    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    test_start_time = time.time()
    dist.launch(train_vqvae, args.n_gpu, 1, 0, args.dist_url, args=(args,))
    test_end_time = time.time()
    print(f'Test time: {(test_end_time - test_start_time):.2f} seconds.')

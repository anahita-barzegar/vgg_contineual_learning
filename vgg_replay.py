import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time

from sklearn.utils import shuffle


class ShuffleDataset(Dataset):
    def __init__(self, x_set, y_set, is_numpy):
        self.x = x_set
        if is_numpy:
            self.y = np.array(y_set)
            self.shuffle_order = np.arange(len(self.x))
            np.random.shuffle(self.shuffle_order)
            self.x_shuf = self.x[self.shuffle_order[0]]
            self.y_shuf = self.y[self.shuffle_order[0]]
        else:
            self.x_shuf, self.y_shuf = shuffle(x_set, y_set, random_state=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x_shuf[idx], self.y_shuf[idx]

    def on_epoch_end(self):
        np.random.shuffle(self.shuffle_order)
        self.x_shuf = self.x[self.shuffle_order]
        self.y_shuf = self.y[self.shuffle_order]

# # # Usage
# # x_train = np.asarray(['path/to/file1.npy', 'path/to/file2.npy', ...])
# # y_train = np.asarray([0, 1, ...])
# # x_val = np.asarray(['path/to/val_file1.npy', 'path/to/val_file2.npy', ...])
# # y_val = np.asarray([0, 1, ...])
#
# train_dataset = ShuffleDataset(x_train, y_train)
# val_dataset = ShuffleDataset(x_val, y_val)
#
# batch_size_train = 32
# batch_size_val = 32
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
#
# # Example of how to use the data loader in a training loop
# for epoch in range(num_epochs):
#     # Training
#     for batch in train_loader:
#         images, labels = batch
#         # Train your model here
#
#     # Call on_epoch_end at the end of each epoch
#     train_dataset.on_epoch_end()
#     val_dataset.on_epoch_end()

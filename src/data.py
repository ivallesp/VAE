import torchvision
import torch
import numpy as np


def get_mnist_batcher(batch_size):
    """Downloads MNIST and stores in the data folder in case it is not available, and
    builds a data loader based batcher of the specified size

    Args:
        batch_size (int): size of the minibatch

    Returns:
        torch data_loader: iterator that builds minibatches of the specified batch
    """

    def transform_mnist(x):
        return np.array(x).astype(np.float32).reshape(-1) / 255

    data = torchvision.datasets.MNIST("data", download=True, transform=transform_mnist)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader

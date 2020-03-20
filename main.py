import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data import get_mnist_batcher
from src.architecture import BasicVAE
from src.plot import build_collage
from src.model import run_epoch


N_EPOCHS = 300
BATCH_SIZE = 64
LATENT_SPACE_SIZE = 2


def main():
    # Load the data
    batcher = get_mnist_batcher(batch_size=BATCH_SIZE)

    # Initialize the model
    vae = BasicVAE(latent_size=LATENT_SPACE_SIZE)

    # Train the model N_EPOCH epochs
    epoch_losses = []
    for epoch in tqdm(range(N_EPOCHS)):
        epoch_loss = run_epoch(vae, batcher)
        epoch_losses.append(epoch_loss)

    # Plot the loss curve
    plt.figure(figsize=[14, 7])
    plt.plot(epoch_losses)
    plt.title("Compounded loss function")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join("img", "loss_mnist.png"))

    # Plot some samples and their respective generations
    x, y = iter(batcher).next()
    x_hat = vae(x)[2]

    images = x.cpu().data.numpy().reshape(-1, 28, 28) * 256
    reconstructions = x_hat.cpu().data.numpy().reshape(-1, 28, 28) * 256

    images_collage = build_collage(images, 8, 8)
    reconstructions_collage = build_collage(reconstructions, 8, 8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[14, 7])
    ax1.imshow(images_collage, cmap="gray_r")
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_title("Original")
    ax2.imshow(reconstructions_collage, cmap="gray_r")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_title("Reconstruction")
    plt.savefig(os.path.join("img", "sample_mnist.png"))

    # Encode all the images into the latent space and plot them
    zs = []
    labels = []
    for i, (x, y) in enumerate(batcher):
        mu, sigma = vae.encoder.forward(x)
        e = torch.randn_like(mu)
        z = mu + sigma * e
        z = z.data.numpy()
        y = y.data.numpy()
        zs.append(z)
        labels.append(y)
    zs = np.concatenate(zs, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=[16, 12])
    plt.scatter(zs[:, 0], zs[:, 1], s=1, alpha=0.7, c=labels)
    plt.colorbar()
    plt.savefig(os.path.join("img", "projection_mnist.png"))

    # Decode a set of points in grid over the latent space and plot the reconstructions
    generations = []
    for zx in np.linspace(-3, 3, 30):
        for zy in np.linspace(-3, 3, 30):
            z = torch.from_numpy(np.array([zx, zy]).astype(np.float32)[None])
            gen = vae.decoder(z).reshape(28, 28).data.numpy()
            generations.append(gen)
    images_collage = build_collage(generations, 30, 30)
    plt.figure(figsize=[7, 7])
    plt.imshow(images_collage, cmap="gray_r")
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join("img", "generation_mnist.png"))


if __name__ == "__main__":
    main()

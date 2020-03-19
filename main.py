from src.data import get_mnist_batcher
from src.architecture import BasicVAE
import matplotlib.pyplot as plt
from src.plot import build_collage

if __name__ == "__main__":
    batcher = get_mnist_batcher(batch_size=64)

    vae = BasicVAE(latent_size=2)

    for epoch in range(100):
        for x, y in batcher:
            loss, x_hat = vae.step(x)

    images = x.cpu().data.numpy().reshape(-1, 28, 28) * 256
    reconstructions = x_hat.cpu().data.numpy().reshape(-1, 28, 28) * 256

    print(images.shape)
    print(reconstructions.shape)

    images_collage = build_collage(images, 8, 8)
    reconstructions_collage = build_collage(reconstructions, 8, 8)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(images_collage)
    ax2.imshow(reconstructions_collage)
    plt.show()

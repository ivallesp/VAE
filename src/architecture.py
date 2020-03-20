import torch


class BasicVAE(torch.nn.Module):
    def __init__(self, latent_size, lr=1e-3):
        super().__init__()
        input_size = 784
        self.encoder = BasicEncoder(input_size, latent_size)
        self.decoder = BasicDecoder(latent_size, input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        e = torch.randn_like(mu)
        z = mu + sigma * e
        x_hat = self.decoder(z)
        return mu, sigma, x_hat

    def calculate_loss(self, x):
        mu, sigma, x_hat = self.forward(x)

        # Reconstruction error
        eps = 1e-9
        rec_error = x * torch.log(x_hat + eps) + (1 - x) * torch.log(1 - x_hat + eps)
        rec_error = -rec_error.sum(axis=1, keepdim=True)

        # KL-divergence error
        kl_error = 1 + sigma - mu.pow(2) - torch.exp(sigma)
        kl_error = -0.5 * kl_error.sum(axis=1, keepdim=True)

        # Compute total error
        total_error = rec_error + kl_error

        return total_error, x_hat

    def step(self, x):
        loss, x_hat = self.calculate_loss(x)
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
        return loss, x_hat


class BasicEncoder(torch.nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.h1 = torch.nn.Linear(in_features=input_size, out_features=128)

        self.mu = torch.nn.Linear(in_features=128, out_features=latent_size)
        self.sigma = torch.nn.Linear(in_features=128, out_features=latent_size)

    def forward(self, x):
        h = torch.nn.functional.relu(self.h1(x))
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu, sigma


class BasicDecoder(torch.nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.h1 = torch.nn.Linear(in_features=latent_size, out_features=128)
        self.out = torch.nn.Linear(in_features=128, out_features=output_size)

    def forward(self, x):
        h = torch.nn.functional.relu(self.h1(x))
        output = torch.sigmoid(self.out(h))
        return output

import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class vae(nn.Module):
    def __init__(self,num_feats,hidden_sizes_encoder,hidden_sizes_decoder,out_size):
        # encoder
        self.hidden_sizes_encoder = [num_feats] + hidden_sizes_encoder + [32]
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes_encoder):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes_encoder[idx],
                                         out_channels=self.hidden_sizes_encoder[idx + 1],
                                         kernel_size=3, stride=1, bias=False, padding=1))
            #self.layers.append(nn.BatchNorm2d(channel_size))
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(Flatten())
        self.encoder = nn.Sequential(*self.layers)
        self.mu = nn.Linear(hidden_sizes_encoder[-2], hidden_sizes_encoder[-1])
        self.log_var = nn.Linear(hidden_sizes_encoder[-2], hidden_sizes_encoder[-1])

        #decoder
        self.layers = []
        self.hidden_sizes_decoder = hidden_sizes_encoder +[out_size]
        self.fc = nn.Linear(hidden_sizes_encoder[-1], hidden_sizes_decoder[0])
        self.layers.append(Unflatten())
        for idx, channel_size in enumerate(hidden_sizes_encoder):
            self.layers.append(nn.ConvTranspose2d(self.hidden_sizes_decoder[idx],self.hidden_sizes_decoder[idx+1],
                                                  kernel_size=5,stride=2))
            self.layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*self.layers)

    def reparametrize(self,mu,log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def encoder(self,x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self.reparametrize(mu,log_var)
        return z,mu,log_var

    def decoder(self,z):
        z = self.fc(z)
        z = self.decoder(z)
        return z

    def forward(self,x):
        z,mu,log_var = self.encoder(x)
        z = self.decoder(z)
        return z,mu,log_var

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD





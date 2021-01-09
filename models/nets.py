import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# https://github.com/pytorch/examples/blob/master/vae/main.py
class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()

        assert isinstance(params.encoder_layers[-1], int), \
                'params.encoder_layers[-1]={}, \
                but the dimension of the last encoder layer must be an integer'
        self.encoder_except_last_layer, in_dim_encoder_last_layer = \
                self._make_layers(params.input_dim, params.encoder_layers[:-1])
        self.encoder_last_layer_1 = nn.Linear(in_dim_encoder_last_layer, params.encoder_layers[-1])
        self.encoder_last_layer_2 = nn.Linear(in_dim_encoder_last_layer, params.encoder_layers[-1])
        
        self.decoder, _ = self._make_layers(params.latent_dim, params.decoder_layers)

    def encode(self, x):
        h = self.encoder_except_last_layer(x)
        return self.encoder_last_layer_1(h), self.encoder_last_layer_2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def _make_layers(self, input_dim, layer_dim_list):
        layers = []
        in_dim = input_dim
        for layer_dim in layer_dim_list:
            if layer_dim == 'R':
                layers.append(nn.ReLU())
            elif layer_dim == 'S':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(in_dim, layer_dim))
                in_dim = layer_dim
        return nn.Sequential(*layers), in_dim # returning in_dim is useful for encoder
        

# https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD




if __name__ == '__main__':
    # Test for class `VAE`
    import torch
    import sys 
    sys.path.append(".") 
    import utils

    params = utils.Params('./experiments/mnist_vae/params.json')
    model = VAE(params)
    print(model)
    x = torch.randn(2,3,32,32)
    print(x)
    y = model(x)
    print(y)
    print(y.size())
    



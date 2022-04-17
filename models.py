from torch import nn
import torch

'''
    Discriminator
'''
class Discriminator(nn.Module):
  def __init__(self, latent_space_dim):
    super().__init__()
    self.latent_space_dim = latent_space_dim
    layers = [nn.Linear(2*2*latent_space_dim, latent_space_dim),
              nn.BatchNorm1d(latent_space_dim),
              nn.Dropout(0.3),
              nn.ReLU(),
              nn.Linear(latent_space_dim, 32),
              nn.Dropout(0.3),
              nn.ReLU(),
              nn.Linear(32,10)]
    self.model = nn.Sequential(*layers)
    
  def forward(self, z_s):
    batch_size = z_s.shape[0]
    z_s = z_s.view(batch_size,self.latent_space_dim*2*2)
    return self.model(z_s)

'''
    Fader Network
'''
class FaderNetwork(nn.Module):
    def __init__(self, latent_space_dim, in_channels, attribute_dim):
        super().__init__()
        enc_layers = [nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2), nn.Dropout(0.3),
                  nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2), nn.Dropout(0.3),
                  nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2), nn.Dropout(0.3),
                  nn.Conv2d(256, latent_space_dim, kernel_size=3, stride=2, padding=1)]
                  
        dec_layers = [nn.ConvTranspose2d(latent_space_dim+attribute_dim, 128, 3, 2, 1, 1, bias=False),
                  nn.ReLU(), nn.Dropout(0.3),
                  nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=False),
                  nn.ReLU(), nn.Dropout(0.3),
                  nn.ConvTranspose2d(64, 16, 3, 2, 1, 1, bias=False),
                  nn.ReLU(), nn.Dropout(0.3),
                  nn.ConvTranspose2d(16, 1, 3, 2, 1,1, bias=False),
                  nn.Tanh()]
                  
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, images, attr):
        enc = self.encode(images)
        return self.decode(enc, attr)

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z, attr):
        z_s = torch.cat([z, attr], dim=1)
        
        
        # for l in self.dec_layers:
        #     if type(l) == nn.ConvTranspose2d:
        #         attr = torch.cat([attr, attr], dim=2)
        #         attr = torch.cat([attr, attr], dim=3)
        #         out = torch.cat([out, attr], dim=1)
            
        #     out = l(out)

        return self.decoder(z_s)
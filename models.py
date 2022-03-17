import numpy as np
import time,os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(32,2))
    self.linears = nn.Sequential(
        nn.Linear(512*2*2, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.Dropout(0.3),
        nn.ReLU()
    )
    
  def forward(self, z_s):
    batch_size = z_s.shape[0]
    z_s = z_s.view(batch_size,512*2*2)
    maps = self.linears(z_s)
    preds = self.layer1(maps)
    preds = nn.Sigmoid()(preds)
    return maps,preds
    
class FaderNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
    self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))



    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(512+2, 512, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv2 = nn.Sequential(
        nn.ConvTranspose2d(512+2, 256, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv3 = nn.Sequential(
        nn.ConvTranspose2d(256+2, 128, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv4 = nn.Sequential(
        nn.ConvTranspose2d(128+2, 64, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv5 = nn.Sequential(
        nn.ConvTranspose2d(64+64+2, 32, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv6 = nn.Sequential(
        nn.ConvTranspose2d(32+32+2, 16, 4, 2, 1, bias=False),
        nn.ReLU()
        )
    self.conv7 = nn.Sequential(
        nn.ConvTranspose2d(16+16+2, 3, 4, 2, 1, bias=False),
        nn.ReLU()
        )

      
  def forward(self, imgs, labels):
    batch_size = imgs.shape[0]

    z_s,skip1,skip2,skip3 = self.encode(imgs) #pass the skip connections over to the decoder
    
    reconsts = self.decode(z_s, labels,skip1,skip2,skip3)
    
    return reconsts
 
  def encode(self, imgs):
    out1 = self.layer1(imgs)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)

    return out7,out1,out2,out3
  
  def decode_prob(self, z_s, hot_labels,skip1,skip2,skip3):
    z_s = torch.cat([z_s, hot_labels], dim=1)

    out1 = self.conv1(z_s)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2) #expand the label vector to concatenate with intermediate outputs
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out1 = torch.cat([out1, hot_labels], dim=1)
    
    out2 = self.conv2(out1)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out2 = torch.cat([out2, hot_labels], dim=1)
    
    out3 = self.conv3(out2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    out3 = torch.cat([out3, hot_labels], dim=1)

    out4 = self.conv4(out3)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out4 = torch.cat([out4, hot_labels], dim=1)
    out4 = torch.cat([out4, skip3], dim=1)
    out4 = torch.cat([out4, hot_labels], dim=1)

    out5 = self.conv5(out4)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out5 = torch.cat([out5, hot_labels], dim=1)
    out5 = torch.cat([out5, skip2], dim=1)
    out5 = torch.cat([out5, hot_labels], dim=1)

    out6 = self.conv6(out5)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=2)
    hot_labels = torch.cat([hot_labels, hot_labels], dim=3)
    # out6 = torch.cat([out6, hot_labels], dim=1)
    out6 = torch.cat([out6, skip1], dim=1)
    out6 = torch.cat([out6, hot_labels], dim=1)

    out7 = self.conv7(out6)

    return out7
  def decode(self, z_s, labels,skip1,skip2,skip3):
    batch_size = len(labels)
    hot_digits = torch.zeros((batch_size, 2, 2, 2)).to(device)
    labels = labels.long()
    for i, digit in enumerate(labels):
      hot_digits[i,digit,:,:] = 1
    
    return self.decode_prob(z_s, hot_digits,skip1,skip2,skip3)


fader = FaderNetwork().to(device)
disc = Discriminator().to(device)
fader.cuda()
disc.cuda()


    
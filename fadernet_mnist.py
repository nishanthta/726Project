import numpy as np
import time,os
from argparse import ArgumentParser
import torch
import torchvision
# import umap
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.utils.data as utils
from tqdm import tqdm
from pyutils import population_mean_norm,show
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from models_mnist import fader, disc
from torchsummary import summary
# from mnist import MNIST

# summary(fader, (1, 28, 28))

# exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use GPU if available

parser = ArgumentParser(description = "customize training")
parser.add_argument('--disc_schedule', '-ds', default = '0.00001')
parser.add_argument('--fader_lr', '-f', default = '0.002')
parser.add_argument('--disc_lr', '-d', default = '0.00002')
args = parser.parse_args()

# -ds 0.003 --fader_lr 0.001 --disc_lr 0.001

#Normalize with population mean and standard deviation

#pop_mean, pop_std0 = population_mean_norm(path = "../select")

# train_dataset = torchvision.datasets.ImageFolder(
#         root="../mnist",

#         transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=pop_mean, std=pop_std0)
#                 ])

#     )

# load mnist
train_dataset = torchvision.datasets.MNIST(
    root="../mnist/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)



train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, 
        num_workers=0,
        shuffle=True
    )
# print(fader)


#TRAIN/TEST


umap_data1 = [] #list of latent space representations for plotting UMAPs
umap_labels = []
disc_data = []

fader_optim = optim.Adam(fader.parameters(), lr=float(args.fader_lr), betas=(0.5,0.999))
disc_optim = optim.Adam(disc.parameters(), lr=float(args.disc_lr), betas=(0.5,0.999))

def train(epoch):
    fader.train() #set to train mode
    disc.train()

    sum_disc_loss = 0
    sum_disc_acc = 0
    sum_rec_loss = 0
    sum_fader_loss = 0
    disc_weight = 0
    disc_weight = 0.0003 + epoch*float(args.disc_schedule) #Use as a knob for tuning the weight of the discriminator in the loss function

    for data, labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
        data = data.to(device)
        labels = labels.long().to(device)
        
        # Encode data
        z,skip1,skip2,skip3 = fader.encode(data)

        if epoch%10 == 0 and z.shape[0] == 32:
            z_temp = z.view(z.shape[0], 512*2*2)
            z_temp = z_temp.cpu()
            temp = z_temp.detach().numpy() 
            labs = labels.cpu().detach().numpy()
            umap_data1.append(temp[1]) 
            umap_labels.append(labs[1])

        
        # Train discriminator
        disc_optim.zero_grad()        
        maps,label_probs = disc(z)
        disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
        sum_disc_loss += disc_loss.item()
        disc_loss.backward()
        disc_optim.step()

        if epoch%10 == 0 and z.shape[0] == 32:
            temp = maps.cpu().detach().numpy() 
            disc_data.append(temp[1]) 

        # Compute discriminator accuracy
        disc_pred = torch.argmax(label_probs, 1)
        disc_acc = torch.sum(disc_pred == labels)
        sum_disc_acc += disc_acc.item()
        
        
        # Train Fader
        fader_optim.zero_grad()
        z,skip1,skip2,skip3 = fader.encode(data)
        
        # Invariance of latent space from new disc
        _,label_probs = disc(z)
        
        # Reconstruction
        reconsts = fader.decode(z, labels,skip1,skip2,skip3)
        rec_loss = F.mse_loss(reconsts, data, reduction='mean')
        sum_rec_loss += rec_loss.item()
        fader_loss = rec_loss - F.cross_entropy(label_probs, labels, reduction='mean')
        # fader_loss = rec_loss - disc_weight * F.cross_entropy(label_probs, labels, reduction='mean')

        fader_loss.backward()
        fader_optim.step()
        
        sum_fader_loss += fader_loss.item()        
        
    train_size = len(train_loader.dataset)


    # if epoch%10 == 0:
    #     plt.clf()
    #     standard_embedding = umap.UMAP(random_state=42).fit_transform(umap_data1)
    #     plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=umap_labels, s=0.1, cmap='Spectral')
    #     plt.savefig('results/umapresults_fader'+str(epoch)+'.png')
    #     plt.clf()
    #     standard_embedding = umap.UMAP(random_state=42).fit_transform(disc_data)
    #     plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=umap_labels, s=0.1, cmap='Spectral')
    #     plt.savefig('results/umapresults_disc'+str(epoch)+'.png')
    #     umap_data1.clear()
    #     umap_labels.clear()
    #     disc_data.clear()


    print('\nDisc Weight: {:.8f} | Fader Loss: {:.8f} | Rec Loss: {:.8f} | Disc Loss, Acc: {:.8}, {:.8f}'
          .format(disc_weight, sum_fader_loss/train_size, sum_rec_loss/train_size, 
        sum_disc_loss/train_size, sum_disc_acc/train_size), flush=True)
    
    return sum_rec_loss/train_size, sum_disc_acc/train_size, disc_weight

def test(epoch):
    fader.eval() #set to eval mode
    disc.eval()
    rec_losses = 0
    disc_losses = 0
    disc_accs = 0
    rec_accs = 0
    flag = 0
    with torch.no_grad():
        for data_batch, labels in train_loader:
            labels = labels.long().to(device)   
            data_batch = data_batch.to(device)
            z,skip1,skip2,skip3 = fader.encode(data_batch) #record the skip connections to pass over to the decoder

            _,label_probs = disc(z)
            disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
            reconsts = fader(data_batch, labels)
            rec_loss = F.mse_loss(reconsts, data_batch, reduction='mean')
            disc_pred = torch.argmax(label_probs, 1)
            disc_acc = torch.sum(disc_pred == labels)   

            disc_losses += disc_loss.item()
            rec_losses += rec_loss.item()
            disc_accs += disc_acc.item()
          
            data_batch = data_batch[:1].to(device)
            labels = labels[:1].to(device)

            batch_z,skip1,skip2,skip3= fader.encode(data_batch)

            con1 = torch.cat((batch_z, batch_z), 0)
            skip11 = torch.cat([skip1, skip1], 0) 
            skip22 = torch.cat([skip2, skip2], 0)
            skip33 = torch.cat([skip3, skip3], 0)
            con2 = torch.cat((con1, batch_z), 0)
            con3 = torch.cat((con2, batch_z), 0)

            faders = (torch.tensor([0,1]).long()).to(device)

            '''
            KEYS
            0 - original
            1 - reconstruction with original attributes
            2 - reconstruction with modified attributes
            '''

            plt.clf()

            show(make_grid(data_batch.detach().cpu()), 'Epoch {} Original'.format(epoch),epoch,0)

            reconst = fader.decode(batch_z,labels,skip1,skip2,skip3).cpu()
            show(make_grid(reconst.view(1, 1, 28, 28)), 'Epoch {} Reconst with Orig Attr'.format(epoch),epoch,1)

            hot_digits = torch.zeros((batch_z.shape[0], 10, 2, 2)).to(device)
            hot_digits[:,3,:,:] = 1

            fader_reconst = fader.decode(batch_z,hot_digits,skip1,skip2,skip3).cpu()
            show(make_grid(fader_reconst.view(1, 1, 28, 28), nrow=2), 'Epoch {} Reconst With Attr 3'.format(epoch),epoch,2)
            break

        print('Test Rec Loss: {:.8f}'.format(rec_losses / len(train_loader.dataset)))
        print('Test disc Loss: {:.8f}'.format(disc_losses / len(train_loader.dataset)))
        print('Test disc accs: {:.8f}'.format(disc_accs / len(train_loader.dataset)))

epochs = 100

recs, accs, disc_wts = [], [], []
for epoch in range(epochs):
    rec_loss, disc_acc, disc_wt = train(epoch)
    recs.append(rec_loss)
    accs.append(disc_acc)
    disc_wts.append(disc_wt)
    
    if epoch % 10 == 0:
        test(epoch)
        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1)
        plt.title('Disc Weight')
        plt.plot(disc_wts)
        plt.subplot(1,3,2)
        plt.title('Reconst Loss')
        plt.plot(recs)
        plt.subplot(1,3,3)
        plt.title('Disc Acc')
        plt.plot(accs)
        plt.savefig('results/losses'+str(epoch)+'.png')
        torch.save(fader.state_dict(), 'results/fader'+str(epoch)+'.pt')


plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
plt.title('Disc Weight')
plt.plot(disc_wts)
plt.subplot(1,3,2)
plt.title('Reconst Loss')
plt.plot(recs)
plt.subplot(1,3,3)
plt.title('Disc Acc')
plt.plot(accs)

plt.savefig('plots.png')
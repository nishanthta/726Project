import torch
from torch import optim
from torch.nn import functional as F
from models import Discriminator, FaderNetwork
from argparse import ArgumentParser
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyutils import show
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use GPU if available

parser = ArgumentParser(description = "customize training")
parser.add_argument('--disc_schedule', '-ds', default = '0.000001')
parser.add_argument('--fader_lr', '-f', default = '0.0002')
parser.add_argument('--disc_lr', '-d', default = '0.0002')
parser.add_argument('--latent_space_dim', default =256)
parser.add_argument('--in_channel', default=1)
parser.add_argument('--attr_dim', default =10)
parser.add_argument('--print_every', default =10)
parser.add_argument('--data', default = 'mnist')
args = parser.parse_args()


# Load Data

train_dataset = None
if args.data == "mnist":
    train_dataset = torchvision.datasets.MNIST(
    root="../mnist/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
else:
    train_dataset = torchvision.datasets.ImageFolder(
        root="../select",
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, 
        num_workers=0,
        shuffle=True
    )

# Models
fader = FaderNetwork(args.latent_space_dim, args.in_channel, args.attr_dim)
disc = Discriminator(args.latent_space_dim)

#TRAIN/TEST

fader_optim = optim.Adam(fader.parameters(), lr=float(args.fader_lr), betas=(0.5,0.999))
disc_optim = optim.Adam(disc.parameters(), lr=float(args.disc_lr), betas=(0.5,0.999))

def train(epoch):
    fader.train() #set to eval mode
    disc.train()

    sum_disc_loss = 0
    sum_disc_acc = 0
    sum_rec_loss = 0
    sum_fader_loss = 0
    disc_weight = 0
    disc_weight = epoch*float(args.disc_schedule) #Use as a knob for tuning the weight of the discriminator in the loss function


    for data, labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
        data = data.to(device)
        labels = labels.long().to(device)
        
        # Encode data
        z = fader.encode(data)
        
        # Train discriminator     
        label_probs = disc(z)
        disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
        sum_disc_loss += disc_loss.item()

        disc_optim.zero_grad()
        disc_loss.backward()
        disc_optim.step()

        # Compute discriminator accuracy
        disc_pred = torch.argmax(label_probs, 1)
        disc_acc = torch.sum(disc_pred == labels)
        sum_disc_acc += disc_acc.item()
        
        
        # Train Fader
        z = fader.encode(data)
        
        # Invariance of latent space from new disc
        label_probs = disc(z)

        # Prepare attributes
        batch_size = len(labels)
        hot_digits = torch.zeros((batch_size, 10, 2, 2)).to(device)
        labels = labels.long()
        for i, digit in enumerate(labels):
            hot_digits[i,digit,:,:] = 1
        
        # Reconstruction
        reconsts = fader.decode(z, hot_digits)
        rec_loss = F.mse_loss(reconsts, data, reduction='mean')
        sum_rec_loss += rec_loss.item()
        fader_loss = rec_loss - disc_weight * F.cross_entropy(label_probs, labels, reduction='mean')

        fader_optim.zero_grad()
        fader_loss.backward()
        fader_optim.step()
        
        sum_fader_loss += fader_loss.item()        
        
    train_size = len(train_loader.dataset)

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
    
    with torch.no_grad():
        for data_batch, labels in train_loader:
            # Encode batch
            labels = labels.long().to(device)   
            data_batch = data_batch.to(device)
            z = fader.encode(data_batch)

            # Prepare attributes
            batch_size = len(labels)
            hot_digits = torch.zeros((batch_size, 10, 2, 2)).to(device)
            labels = labels.long()
            for i, digit in enumerate(labels):
                hot_digits[i,digit,:,:] = 1

            # Reconstruct
            label_probs = disc(z)
            disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')

            reconsts = fader(data_batch, hot_digits)
            rec_loss = F.mse_loss(reconsts, data_batch, reduction='mean')

            disc_pred = torch.argmax(label_probs, 1)
            disc_acc = torch.sum(disc_pred == labels)   

            disc_losses += disc_loss.item()
            rec_losses += rec_loss.item()
            disc_accs += disc_acc.item()
          
            data_batch = data_batch[:1].to(device)
            labels = labels[:1].to(device)

            batch_z = fader.encode(data_batch)

            '''
            KEYS

            0 - original
            1 - reconstruction with original attributes
            2 - reconstruction with modified attributes

            '''

            plt.clf()

            show(make_grid(data_batch.detach().cpu()), 'Epoch {} Original'.format(epoch),epoch,"img")

            reconst = fader.decode(z, hot_digits).cpu()
            show(make_grid(reconst), 'Epoch {} Reconst with Orig Attr'.format(epoch),epoch,"orig")

            mod_attr = torch.zeros((z.shape[0], 10, 2, 2)).to(device)
            mod_attr[:,3,:,:] = 1

            fader_reconst = fader.decode(z,mod_attr).cpu()
            show(make_grid(fader_reconst), 'Epoch {} Reconst With Attr 3'.format(epoch),epoch,"mod")
            break

        print('Test Rec Loss: {:.8f}'.format(rec_losses / len(train_loader.dataset)))
        print('Test disc Loss: {:.8f}'.format(disc_losses / len(train_loader.dataset)))
        print('Test disc accs: {:.8f}'.format(disc_accs / len(train_loader.dataset)))

epochs = 1001 

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


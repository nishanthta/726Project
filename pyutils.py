import numpy as np
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def population_mean_norm(path): #Utility function to normalize input data based on mean and standard deviation of the entire dataset
    train_dataset1 = torchvision.datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                    transforms.ToTensor()
                    ])
        )

    dataloader = torch.utils.data.DataLoader(train_dataset1, batch_size=4096, shuffle=False, num_workers=4)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for data,label in dataloader:
        numpy_image = data.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0


def show(img, title, epoch, orig): #Utility function to show figures and plots
    npimg = img.numpy()
    plt.figure()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('results/results_'+orig+"_"+str(epoch)+'.png')


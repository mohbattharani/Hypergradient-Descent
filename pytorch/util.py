import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
import torch

def save_plot (logs, model_name, folder ='results'):

    if (not os.path.exists(folder)):
        os.mkdir(folder)
    save_path = os.path.join (folder, model_name)
    if (not os.path.exists (save_path)):
        os.mkdir (save_path)
    
    keys = logs.keys()
    plt.figure()
    for k in keys:
        # normalize so that visulization is readable.
        plt.plot(logs[k], label=k)
    plt.legend()
    plt.savefig (os.path.join( save_path, model_name+'.jpg'))
    


def save_csv (logs, model_name, folder ='results'):
    if (not os.path.exists(folder)):
        os.mkdir(folder)
    save_path = os.path.join (folder, model_name)
    if (not os.path.exists (save_path)):
        os.mkdir (save_path)
    
    df = pd.DataFrame(logs) 

    df.to_csv(os.path.join( save_path, model_name+'.csv'))
    

def data_loader (batch_size, name= 'mnist'):
    if (name == 'cifar10'):
        return cifar10(batch_size)
    else:
        return mnist (batch_size)
    
def cifar10 (batch_size,  transform=transforms.ToTensor()):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def mnist (batch_size, transform=transforms.ToTensor()):
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    

    


    

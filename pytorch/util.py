import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
import torch

def save_plot (logs, model_name, file_name, folder ='results'):
    
    if (not os.path.exists(folder)):
        os.mkdir(folder)
    folder = os.path.join (folder, model_name) 
    if (not os.path.exists(folder)):
        os.mkdir(folder)

    save_path = os.path.join (folder,  model_name+"_"+file_name)
    
    keys = logs.keys()
    plt.figure()
    for k in keys:
        # normalize so that visulization is readable.
        plt.plot(logs[k], label=k)
    plt.legend()
    plt.savefig (os.path.join( save_path+'.jpg'))
    


def save_csv (logs, model_name, file_name, folder ='results'):

    if (not os.path.exists(folder)):
        os.mkdir(folder)
    folder = os.path.join (folder, model_name)
    if (not os.path.exists(folder)):
        os.mkdir(folder) 
   
    save_path = os.path.join (folder, model_name+"_"+file_name)
    #if (not os.path.exists (save_path)):
    #    os.mkdir (save_path)
    
    df = pd.DataFrame(logs) 

    df.to_csv(os.path.join( save_path+'.csv'))
    

def data_loader (batch_size, dataset_name):
    
    name = dataset_name["name"]
    image_size =  dataset_name['input_dim'][-1]
    
 
    if (name == 'cifar10'):
        return cifar10(batch_size, image_size = image_size)
    elif (name == 'mnist'):
        return mnist (batch_size, image_size = image_size)
    elif (name == 'cifar100'):
        return cifar100(batch_size, image_size = image_size)
    
def cifar10 (batch_size,  image_size = 32):
    # https://github.com/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb
    means = [0.485, 0.456, 0.406]
    stds= [0.229, 0.224, 0.225]
    
    
    train_transforms = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(image_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])

    test_transforms = transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds) 
                        ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def mnist (batch_size, image_size = 28):
    transform=transforms.ToTensor()
    if (image_size == None):
        image_size = 28
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    

def cifar100 (batch_size,  image_size = 32):
    # https://github.com/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb
    means = [0.485, 0.456, 0.406]
    stds= [0.229, 0.224, 0.225]
    if (image_size == None):
        image_size = 32
    train_transforms = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(image_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])

    test_transforms = transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds)
                        ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader    


    

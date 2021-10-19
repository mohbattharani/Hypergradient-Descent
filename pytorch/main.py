# Code adapted from https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets

from models import LogisticRegression
from optimizers import SGD, SGDHD
from tqdm import tqdm
from util import save_plot, save_csv

# Load Dataset

def mnist ():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    # Make Dataset Iterable
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def test (model, test_loader):
    total = 0
    correct = 0
    total_loss = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = model(images)
        total_loss += criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        correct+= (predicted == labels).sum()
    accuracy = 100 * correct/total
    
    return total_loss, accuracy

def train (model, train_loader, optimizer, criterion, epochs, test_freq = 1):
    logs = {
        "train_loss": [],
        "test_loss" : [],
        "test_acc": [],
        "epoch": [],
        "lr": []
    }
    for epoch in tqdm (range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        #if (epoch%test_freq == 0):
        test_loss, test_acc = test (model, test_loader)
        
        logs ["epoch"].append (epoch)
        try:
            logs ["lr"].append (optimizer.lr.item())
        except:
            logs ["lr"].append (optimizer.lr)

        logs ["train_loss"].append (loss.item())
        logs ["test_loss"].append (test_loss.item())
        logs ["test_acc"].append (test_acc.item())
        
    return logs
        




# Initialize constants
batch_size = 16
input_dim = 784
output_dim = 10
lr = 0.1
beta = 0.0001
epoch = 2


models = ['LogisticRegression']
opt_names = ['sgd', 'sgdhd']
dataset_names = ['mnist']

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

for dataset_name in dataset_names:
    if (dataset_name == 'mnist'):
        train_loader, test_loader = mnist()
    criterion = torch.nn.CrossEntropyLoss() 

    for m in models:
        for opt_name in opt_names:
            if m == 'LogisticRegression':
                model = LogisticRegression(input_dim, output_dim)
            if (opt_name == 'sgd'):
                opt = SGD (model, lr)
            else:
                opt = SGDHD (model, lr, beta)
            
            logs = train (model, train_loader, opt, criterion, epoch)
            log_name =  m+'_'+opt_name+'_'+dataset_name
            print ("Logname:", log_name)
            save_plot (logs, log_name)
            save_csv (logs, log_name)
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
        logs ["lr"].append (optimizer.lr.item())
        logs ["train_loss"].append (loss.item())
        logs ["test_loss"].append (test_loss.item())
        logs ["test_acc"].append (test_acc.item())
        
    return logs
        
              
    #print("Iteration: {}. lr: {}, Loss: {}. Accuracy: {}.".format(i, optimizer.lr,  loss.item(), accuracy))



# Initialize constants
batch_size = 2
input_dim = 784
output_dim = 10
lr = 0.1
beta = 0.0001

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

model = LogisticRegression(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss() 
sgdhd = SGDHD (model, lr, beta)
sgd = SGD (model, lr)
epoch = 2

train_loader, test_loader = mnist()
logs = train (model, train_loader, sgdhd, criterion, epoch)
print ("logs:", logs["test_loss"])
save_plot (logs, 'LogisticRegression')
save_csv (logs, 'LogisticRegression')
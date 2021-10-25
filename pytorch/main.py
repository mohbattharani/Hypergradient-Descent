# Code adapted from https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

import torch
from torch.autograd import Variable
from torchvision import datasets

from models import *
from optimizers import *
from tqdm import tqdm
from util import save_plot, save_csv, data_loader


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
    test_loss, test_acc = test (model, test_loader)
    train_loss, train_acc = test (model, train_loader)
    logs = {
        "train_loss": [train_loss],
        "test_loss" : [test_loss],
        "test_acc": [test_acc],
        "epoch": [0],
        "lr": [lr]
    }
    for epoch in tqdm (range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            #images = Variable(images.view(-1, 28 * 28))
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
input_dim = 28 * 28
#{'mnist': (28,28, 1),
#             'cifar10': (32,32,,3) }

output_dim = 10
lr = 0.0001
beta = 0.0001
epoch = 5
momentum = 0.5

models = ['LogisticRegression'] #['mlp', 'LogisticRegression']
opt_names = ['sgdnhd']#, 'sgdn', 'adamhd']#, 'sgdhd', 'sgdn', 'sgdnhd', 'adamhd']
dataset_names = ['mnist'] # ['mnist', 'cifar10']

all_logs = {}

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

for dataset_name in dataset_names:
    train_loader, test_loader = data_loader (batch_size, dataset_name) #mnist()
    criterion = torch.nn.CrossEntropyLoss() 

    for m in models:
        model_logs = {}
        for opt_name in opt_names:
            if m == 'LogisticRegression': 
                model = LogReg( input_dim, output_dim)
            elif (m == 'mlp'):
                model = MLP( input_dim, output_dim)


            if (opt_name == 'sgd'):
                opt = SGD (model, lr)
            elif (opt_name == 'sgdhd'):
                opt = SGDHD (model, lr, beta)
            elif (opt_name == 'sgdn'):
                opt = SGDN (model, lr, momentum)
            elif (opt_name == 'sgdnhd'):
                opt = SGDNHD (model, lr, beta, momentum)
            elif (opt_name == 'adamhd'):
                b1=0.9
                b2=0.999
                eps=10**-8
                opt = AdamHD ( model, lr, beta, b1, b2, eps)
            else:
                print ("Error: Please select proper optimizer.")
                exit()
            
            log_name =  m+'_'+opt_name+'_'+dataset_name
            print ("Logname:", log_name)
            logs = train (model, train_loader, opt, criterion, epoch)
            save_plot (logs, log_name)
            save_csv (logs, log_name)
            model_logs [opt_name+'_test_loss'] = logs['test_loss']
        
        save_plot (model_logs, dataset_name+m)
        save_csv (model_logs, dataset_name+m)
            
        
        
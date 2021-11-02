# Code adapted from https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

import torch
from torch.autograd import Variable

import models
from optimizers import *
from tqdm import tqdm
from util import save_plot, save_csv, data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test (model, test_loader):
    total = 0
    correct = 0
    total_loss = 0
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        if (torch.cuda.is_available()):
            predicted = predicted.detach().cpu()
        correct+= (predicted == labels).sum()
        total+= labels.size(0)
        
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
            images = images.to(device)
            labels = Variable(labels).to(device)
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
output_dim = 10
lr = 0.0001
beta = 0.0001
epoch = 1
momentum = 0.5
b1=0.9
b2=0.999
eps=10**-8

model_ames = ['LogisticRegression', 'mlp'] #['mlp', 'LogisticRegression']
opt_names = ['sgd',]#['sgd', 'sgdhd', 'sgdn', 'sgdnhd', 'adam', 'adamhd']
datasets = [{'name':'mnist', 'input_dim': 28*28}, 
            {'name':'cifar10', 'input_dim': 32*32*3}, 
            ] 
datasets = [{'name':'mnist', 'input_dim': 28*28}]

all_logs = {}

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

for dataset in datasets:
    
    train_loader, test_loader = data_loader (batch_size, dataset["name"]) #mnist()
    criterion = torch.nn.CrossEntropyLoss() 

    for model_name in model_ames:
        model_logs = {}
        for opt_name in opt_names:
            model = models.select_model(model_name,  dataset["input_dim"], output_dim)

            if (opt_name == 'sgd'):
                opt = SGD (model, lr)
            elif (opt_name == 'sgdhd'):
                opt = SGDHD (model, lr, beta)
            elif (opt_name == 'sgdn'):
                opt = SGDN (model, lr, momentum)
            elif (opt_name == 'sgdnhd'):
                opt = SGDNHD (model, lr, beta, momentum)
            elif (opt_name == 'adam'):
                opt = Adam ( model, lr, beta, b1, b2, eps)
            elif (opt_name == 'adamhd'):
                opt = AdamHD ( model, lr, beta, b1, b2, eps)
            else:
                print ("Error: Please select proper optimizer.")
                exit()
            
            log_name =  dataset["name"] +'_' + model_name +'_'+opt_name
            print ("Logname:", log_name)
            logs = train (model, train_loader, opt, criterion, epoch)
            save_plot (logs, log_name)
            save_csv (logs, log_name)
            model_logs [opt_name+'_test_loss'] = logs['test_loss']
        
        save_plot (model_logs, dataset["name"] + model_name)
        save_csv (model_logs, dataset["name"] + model_name)
            
        
        
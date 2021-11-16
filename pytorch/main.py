# Code adapted from https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm

import models
from optimizers import *
from util import save_plot, save_csv, data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test (model, test_loader):
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
         for images, labels in tqdm(test_loader):
             images = images.to(device)
             labels = labels.to(device)
       	     outputs = model(images)
        
       	     total_loss += criterion(outputs, labels)
       	     _, predicted = torch.max(outputs.data, 1)
        
       	     correct+= (predicted == labels).sum()
       	     total+= labels.size(0)
    

    accuracy = 100 * correct/total
    if (total_loss.is_cuda):
        total_loss = total_loss.detach().cpu()
    if (accuracy.is_cuda):
        accuracy = accuracy.detach().cpu()    

    return total_loss, accuracy

def train (model, train_loader, test_loader, optimizer, criterion, epochs, test_freq = 1):
    test_loss, test_acc = test (model, test_loader)
    train_loss, train_acc = test (model, train_loader)
    logs = {
        "train_loss": [train_loss.detach().cpu().numpy() if train_loss.is_cuda else train_loss.item() ],
        "test_loss" : [test_loss.detach().cpu().numpy() if test_loss.is_cuda else test_loss.item()],
        "test_acc": [test_acc.detach().cpu().numpy() if test_acc.is_cuda else test_acc.item()],
        "epoch": [0],
        "lr": [optimizer.lr]
    }
    for epoch in tqdm (range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = Variable(labels).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        #if (epoch%test_freq == 0):
        test_loss, test_acc = test (model, test_loader)
        
    
        logs ["epoch"].append (epoch)
        if torch.is_tensor(optimizer.lr):
           lr = (optimizer.lr.detach().cpu().item() if optimizer.lr.is_cuda else optimizer.lr.item())
        else:
           lr = optimizer.lr
        logs ["lr"].append (lr)

        logs ["train_loss"].append (loss.detach().cpu().numpy() if loss.is_cuda else loss.item())
        logs ["test_loss"].append (test_loss.detach().cpu() if test_loss.is_cuda else test_loss.item())
        logs ["test_acc"].append (test_acc.detach().cpu() if test_acc.is_cuda else test_acc.item())
        
    return logs
        


# Initialize constants
batch_size = 128
output_dim = 10
lr = 0.0001
beta = 0.0001
epoch = 100
momentum = 0.5
b1=0.9
b2=0.999
eps=10**-8

model_names = [#{'name':'wideresnet','input_dim': (3,32,32)},
               {'name':'resnet110','input_dim': (3,32,32)},
               #{'name':'vgg','input_dim': (3,32,32)} 
              ]

# [{'name':'vgg','input_dim': [3,214,214]},
#   {'name':'cnn','input_dim': [3,32,32]}, 
#    {'name':'mlp','input_dim': [3,32,32]},
#      {'name':'LogisticRegression','input_dim': [3,32,32]},
#  ]

#'cnn','vgg',]#,'cnn','vgg', 'LogisticRegression', 'mlp'] 

opt_names = ['sgd', 'sgdhd', 'sgdn', 'sgdnhd', 'adam', 'adamhd']
dataset_names = [{'name':'mnist', 'n_classes': 10}, 
                 {'name':'cifar10', 'n_classes': 10}, 
            ] 
#dataset_names = [{'name':'cifar10', 'input_dim': [3,32,32]}]
dataset_names = [{'name':'cifar10', 'n_classes': 10}]

all_logs = {}

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

for dataset_name in dataset_names:
    
    #train_loader, test_loader = data_loader (batch_size, dataset_name) #mnist()
    criterion = torch.nn.CrossEntropyLoss() 

    for model_name in model_names:
        print ("Training start on model:", model_name['name'])
        model_logs = [{}, {}]
         

        dataset_name['input_dim'] = model_name['input_dim']
        
        train_loader, test_loader = data_loader (batch_size, dataset_name) 
        for opt_name in opt_names:
            model = models.select_model(model_name['name'],  model_name["input_dim"], dataset_name['n_classes'])

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
            
            log_name =  dataset_name["name"] +'_'+opt_name
            print ("Logname:", log_name)
            logs = train (model, train_loader, test_loader, opt, criterion, epoch)
            save_plot (logs, model_name['name'], log_name)
            save_csv (logs, model_name['name'], log_name)
            model_logs[0][opt_name+'_test_loss'] = logs['test_loss']
            model_logs[1][opt_name+'_lr'] = logs['lr']
        
        save_plot (model_logs[0], model_name['name'], dataset_name["name"]+'_test_loss',  xlabel='epochs', ylabel='loss')
        save_csv (model_logs[0], model_name['name'], dataset_name["name"]+ '_test_loss')
        save_plot (model_logs[1], model_name['name'], dataset_name["name"]+"_lr", xlabel='epochs', ylabel='lr')
        save_csv (model_logs[1], model_name['name'], dataset_name["name"]+"_lr")     
        
        

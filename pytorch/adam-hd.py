import torch
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import math 

# Load Dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


# Initialize constants
batch_size = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10

# Make Dataset Iterable
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create Model Class
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# Instantiate Model Class
model = LogisticRegression(input_dim, output_dim)

# Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

# Train Model
iter = 0

grad_alpha_ut = None
beta = 0.0001
lr_rate = 0.001
b1=0.9
b2=0.999
eps=10**-8
grad_prev=[]
 

for param in model.parameters():
  param.velocity=None
  param.momentum=None

for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        outputs = model(images)
       
        loss = criterion(outputs, labels)
        loss.backward()
        grad=[]
        for param in model.parameters():
          gt= param.grad.data 
          grad.append(torch.flatten(gt))
        
        
        grad=torch.cat(grad, 0)
        if (grad_alpha_ut is None):
          grad_prev = torch.zeros_like (grad)
            
        h_t = torch.dot(grad, grad_prev)              ## Hyperparameter gradient
        lr_rate= lr_rate - beta * h_t
        grad_prev=[]

        for param in model.parameters():
            # SGD
            gt = param.grad.data
            if (param.velocity is None):
              param.velocity =  torch.zeros_like (gt)
            if (param.momentum is None):
              param.momentum =  torch.zeros_like (gt)
            
            #print("Gradient", gt)
            #print("Momentum Early", param.momentum)
            #print("Velocity1 Early",param.velocity)

            param.momentum = torch.mul(b1 , param.momentum) + torch.mul ((1- b1) , gt )
            param.velocity = torch.mul(b2, param.velocity)  + (1-b2) * torch.square(gt)
            
            #print("Momentum", param.momentum)
            #print("Velocity",param.velocity)

            momentum_bias = param.momentum/(1-pow(b1,i*epoch+1)) ## epoch  +1 or not?
            velocity_bias = param.velocity/(1-pow(b2,i*epoch+1)) ## epoch  +1 or not?

            #print("Momentum bias", momentum_bias )
            #print("Velocity bias",velocity_bias)

            u_t = - (lr_rate/(math.sqrt(i* epoch +1))) * (momentum_bias)/(torch.sqrt(velocity_bias) + eps)

            #print("Update", u_t) 
            grad_alpha_ut= -1 * (momentum_bias)/(torch.sqrt(velocity_bias) + eps)
            grad_prev.append(torch.flatten(grad_alpha_ut))
            param.data = param.data + u_t
            #print ("ht:{}, lr:{}".format(h_t, lr_rate))

        for param in model.parameters():
          param.grad.zero_()

        grad_prev=torch.cat(grad_prev, 0)

       
        iter+=1

        if iter%50==49:
        #while(iter < 10): 
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
            #print ("ht:{}, lr:{}".format(h_t, lr_rate))


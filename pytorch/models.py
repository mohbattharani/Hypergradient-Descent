import torch 
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def select_model (model_name, input_dim, output_dim):
    if (model_name == 'mlp'):
        model = MLP(input_dim, output_dim)
    elif model_name == "LogisticRegression":
        model = LogReg( input_dim, output_dim)
    elif model_name == 'cnn':
        model = CNN (input_dim, output_dim)
    elif model_name == "vgg":
        model = VGG()

    elif model_name == "Resnet18":
        model = Resnet18()
    elif model_name == "WideResnet":
        model = WideResnet()
        
    else:
        print ("Error!: Model not available. Please select right model name.")
        print ("Available models: mlp, LogisticRegression,VGG,Resnet18,WideResnet")
        return None

    return model.to (device)


# Create Model Class
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = 1
        for d in input_dim:
            self.input_dim *= d
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(self.input_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        outputs = self.linear(x)
        return outputs

class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self.input_dim = 1
        for d in input_dim:
            self.input_dim *= d
        self.lin1 = nn.Linear(self.input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 512):
        super(MLP, self).__init__()
        self.input_dim = 1
        for d in input_dim:
            self.input_dim *= d
        self.lin1 = nn.Linear(self.input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential( nn.Conv2d(input_dim[0], 8, (3,3), (2,2)),  nn.ReLU())  
        self.conv2 = nn.Sequential( nn.Conv2d(8, 16, (3,3), (2,2)),  nn.ReLU())
        self.conv3 = nn.Sequential( nn.Conv2d(16, 16, (3,3), (2,2)),  nn.ReLU())  

        self.out = nn.Linear(16 * input_dim[-1]//16 * input_dim[-1]//16 , output_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization


def VGG ():
    model_v = models.vgg16( pretrained=False)
    return model_v

def Resnet18():
    model_r = models.resnet18(pretrained=False)
    return model_r
    
def WideResnet():
    model_w = models.wide_resnet50_2(pretrained=False)
    return model_w


    



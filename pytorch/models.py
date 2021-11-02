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
        self._input_dim = input_dim
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, self._input_dim)
        outputs = self.linear(x)
        return outputs

class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 1000):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

def VGG ():
    model_v = models.vgg16( pretrained=False)
    return model_v

def Resnet18():
    model_r = models.resnet18(pretrained=False)
    return model_r
    
def WideResnet():
    model_w = models.wide_resnet50_2(pretrained=False)
    return model_w


    



import torch 
import torch.nn.functional as F
import torch.nn as nn

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
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim = 32
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
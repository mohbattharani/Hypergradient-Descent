import torch 

class SGDHD ():
    def __init__(self, model, lr, beta):
        self.model = model
        self.lr = lr 
        self.prev_grad = None
        self.beta = beta

    # SGD-HD
    def step (self):
        grads = []
        for param in self.model.parameters():   
          g_t = param.grad.data   
          u_t = - self.lr * g_t
          param.data = param.data + u_t
          grads.append (g_t.view (-1))
          self.grads = grads
        grads = torch.cat (grads, 0)

        if (self.prev_grad is None):
          self.prev_grad = torch.zeros_like (grads)
        
        h_t = torch.dot (grads, self.prev_grad)
        self.lr = self.lr + self.beta * h_t
        self.prev_grad = grads
        for param in self.model.parameters():
           param.grad.zero_()

class SGD ():
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr 
        self.prev_grad = None

    def step (self):
        grads = []
        for param in self.model.parameters():   
          g_t = param.grad.data   
          u_t = - self.lr * g_t
          param.data = param.data + u_t
          param.grad.zero_()

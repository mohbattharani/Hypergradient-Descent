import torch 

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


class SGDN ():
    def __init__(self, model, lr, momentum):
        self.model = model
        self.lr = lr 
        self.prev_grad = None
        self.momentum = momentum
        for param in self.model.parameters():
          param.velocity=None

    def step (self):
        grads = []
        for param in self.model.parameters(): 
          g_t = param.grad.data     
          if (param.velocity is None):
            param.velocity =  torch.zeros_like (g_t)
          param.velocity = self.momentum * param.velocity + g_t

          u_t = - self.lr * (g_t + self.momentum * param.velocity)
          param.data = param.data + u_t
          param.grad.zero_()

class SGDNHD ():
    def __init__(self, model, lr, beta, momentum):
        self.model = model
        self.lr = lr 
        self.prev_grad = None
        self.beta = beta
        self.momentum = momentum
        for param in self.model.parameters():
          param.velocity=None

    # SGD-HD
    def step (self):
        grads = []
        for param in self.model.parameters():   
          g_t = param.grad.data
          if (param.velocity is None):
            param.velocity =  torch.zeros_like (g_t)
          param.velocity = self.momentum * param.velocity + g_t

          u_t = - self.lr * (g_t + self.momentum * param.velocity)
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

import torch
import copy 
import math 

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
          #print(param.size())
          g_t = param.grad.data   
          u_t = - self.lr * g_t
          param.data = param.data + u_t
          grads.append (g_t.view (-1))
          
        grads = torch.cat (grads, 0)

        if (self.prev_grad is None):
          self.prev_grad = torch.zeros_like (grads)
        
        h_t = torch.dot (grads, self.prev_grad)
        self.lr = self.lr - self.beta * h_t
        self.prev_grad = - grads

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

    # SGDN-HD
    def step (self):
        grads = []
        grads_ut = []
        
        for param in self.model.parameters():   
          g_t = param.grad.data
          if (param.velocity is None):
            param.velocity =  torch.zeros_like (g_t)
          param.velocity = self.momentum * param.velocity + g_t
          u_t = - self.lr * (g_t + self.momentum * param.velocity)
          grad_alpha_ut = -g_t - self.momentum * param.velocity
          
          param.data = param.data + u_t

          grads.append (g_t.view (-1))
          grads_ut.append (grad_alpha_ut.view (-1))
          
        grads = torch.cat (grads, 0)
        grads_ut = torch.cat (grads_ut, 0)

        if (self.prev_grad is None):
          self.prev_grad = torch.zeros_like (grads)
        
        h_t = torch.dot (grads, self.prev_grad)
        self.lr = self.lr - self.beta * h_t
        self.prev_grad = grads_ut
        for param in self.model.parameters():
           param.grad.zero_()


class Adam ():
    def __init__(self, model, lr, beta, b1, b2, eps):
        self.model = model
        self.lr = lr 
        self.lr_global = copy.deepcopy (lr) 
        self.prev_grad = None
        self.beta = beta
        self.b1 = b1 
        self.b2 = b2
        self.eps = eps
        self.t = 0
        for param in self.model.parameters():
          param.velocity=None
          param.momentum=None

    # Adam
    def step (self):
        self.t += 1   # verified from code

        for param in self.model.parameters():   
          g_t = param.grad.data
          
          if (param.velocity is None):
            param.velocity =  torch.zeros_like (g_t)
          if (param.momentum is None):
            param.momentum =  torch.zeros_like (g_t)

          param.momentum = torch.mul(self.b1 , param.momentum) + torch.mul ((1- self.b1) , g_t )
          param.velocity = torch.mul(self.b2, param.velocity)  + (1-self.b2) * torch.square(g_t)

          momentum_bias = param.momentum/(1-pow(self.b1, self.t)) 
          velocity_bias = param.velocity/(1-pow(self.b2, self.t)) 

          u_t = - (self.lr/(math.sqrt(self.t))) * (momentum_bias)/(torch.sqrt(velocity_bias) + self.eps)
          param.data = param.data + u_t
        
        for param in self.model.parameters():
          param.grad.zero_()




class AdamHD ():
    def __init__(self, model, lr, beta, b1, b2, eps):
        self.model = model
        self.lr = lr 
        self.lr_global = copy.deepcopy (lr) 
        self.prev_grad = None
        self.beta = beta
        self.b1 = b1 
        self.b2 = b2
        self.eps = eps
        self.t = 0
        for param in self.model.parameters():
          param.velocity=None
          param.momentum=None

    # Adam-HD
    def step (self):
        grads = []
        grads_ut = []
        self.t = 1   # verified from code

        for param in self.model.parameters():   
          g_t = param.grad.data
          
          if (param.velocity is None):
            param.velocity =  torch.zeros_like (g_t)
          if (param.momentum is None):
            param.momentum =  torch.zeros_like (g_t)

          param.momentum = torch.mul(self.b1 , param.momentum) + torch.mul ((1- self.b1) , g_t )
          param.velocity = torch.mul(self.b2, param.velocity)  + (1-self.b2) * torch.square(g_t)

          momentum_bias = param.momentum/(1-pow(self.b1, self.t)) 
          velocity_bias = param.velocity/(1-pow(self.b2, self.t)) 

          u_t = - (self.lr/(math.sqrt(self.t))) * (momentum_bias)/(torch.sqrt(velocity_bias) + self.eps)

          grad_alpha_ut= -1 * (momentum_bias)/(torch.sqrt(velocity_bias) + self.eps)
          param.data = param.data + u_t
        
          grads.append (g_t.view (-1))
          grads_ut.append (grad_alpha_ut.view (-1))
          
        grads = torch.cat (grads, 0)
        grads_ut = torch.cat (grads_ut, 0)

        if (self.prev_grad is None):
          self.prev_grad = torch.zeros_like (grads)
          
        h_t = torch.dot (grads, self.prev_grad)
        self.lr = self.lr - self.beta * h_t
        self.prev_grad = grads_ut

        for param in self.model.parameters():
          param.grad.zero_()




## COPY provided by huzaifa

class AdamHD2 ():
    def __init__(self, model, lr, beta, b1, b2, eps):
        self.model = model
        self.lr = lr 
        self.lr_global = copy.deepcopy (lr) 
        self.prev_grad = None
        self.beta = beta
        self.b1 = b1 
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.grad_alpha_ut = None
        for param in self.model.parameters():
          param.velocity=None
          param.momentum=None

    # Adam-HD
    def step (self):
      grad=[]
      self.t += 1
      epoch = self.t
      for param in self.model.parameters():
          gt= param.grad.data 
          grad.append(torch.flatten(gt))
        
      grad = torch.cat(grad, 0)

      if (self.grad_alpha_ut is None):
        grad_prev = torch.zeros_like (grad)
              
      h_t = torch.dot(grad, grad_prev)              ## Hyperparameter gradient
      lr_rate= self.lr - self.beta * h_t
      grad_prev=[]

      for param in self.model.parameters():
              # SGD
              gt = param.grad.data
              if (param.velocity is None):
                param.velocity =  torch.zeros_like (gt)
              if (param.momentum is None):
                param.momentum =  torch.zeros_like (gt)
              
              #print("Gradient", gt)
              #print("Momentum Early", param.momentum)
              #print("Velocity1 Early",param.velocity)

              param.momentum = torch.mul(self.b1 , param.momentum) + torch.mul ((1- self.b1) , gt )
              param.velocity = torch.mul(self.b2, param.velocity)  + (1-self.b2) * torch.square(gt)
              
              #print("Momentum", param.momentum)
              #print("Velocity",param.velocity)

              momentum_bias = param.momentum/(1-pow(self.b1,epoch)) ## epoch  +1 or not?
              velocity_bias = param.velocity/(1-pow(self.b2,epoch)) ## epoch  +1 or not?

              #print("Momentum bias", momentum_bias )
              #print("Velocity bias",velocity_bias)

              u_t = - (self.lr/(math.sqrt(epoch))) * (momentum_bias)/(torch.sqrt(velocity_bias) + self.eps)

              #print("Update", u_t) 
              grad_alpha_ut= -1 * (momentum_bias)/(torch.sqrt(velocity_bias) + self.eps)
              grad_prev.append(torch.flatten(grad_alpha_ut))
              param.data = param.data + u_t
              #print ("ht:{}, lr:{}".format(h_t, lr_rate))

      for param in self.model.parameters():
          param.grad.zero_()

      self.grad_prev=torch.cat(grad_prev, 0)


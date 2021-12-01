# Hypergradient-Descent
##### Course Project: Machine Learning & Optimization (Fall-2021)
###### Group: [Mohbat Tharani](http://mohbat.weebly.com/), Momin Abbas, Huzaifa Arif, Tobia Park, Xinyan Sun




Language: Python
API: Pytorch

# Instructions

The pytorch/data folder contains the datasets used for training (MNIST and cifar-10). Results of our experiments are located in the pytorch/results folder. 

To reproduce all of the experiments, simply download all the files in the pytorch folder and run main.py as ``` python main.py ```.


Pytorch optimizers for all of the Hypergradient Descent algorithms are located in pytorch/optimizers.py if you want to utilize these algorithms for your own purposes or to run your own experiments. They are normal pytorch optimizers that can simply be plugged in and used the same as you would use any normal pytorch optimizer.


To select which models, dataset or optimizers you want to use to regenerate the results, just add that dataset in the respcective files. ```model.py``` contains model `classes`, ```util.py``` has codes for dataloaders for datasets including `MNIST`, `CIFAR10` and `CIFAR100`. However, you can use your dataloader for your dataset(s). 


```beta.py``` has the code that generates convergence trends for different values of `beta`. 

# Reference
```
@article{baydin2017online,
  title={Online learning rate adaptation with hypergradient descent},
  author={Baydin, Atilim Gunes and Cornish, Robert and Rubio, David Martinez and Schmidt, Mark and Wood, Frank},
  journal={arXiv preprint arXiv:1703.04782},
  year={2017}
}
```


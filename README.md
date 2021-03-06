# Hypergradient-Descent
##### Course Project: [CSCI-6961/4961 Machine Learning & Optimization (Fall-2021)](https://www.cs.rpi.edu/~gittea/teaching/fall2021/mlandopt.html)
###### Group: [Mohbat Tharani](http://mohbat.weebly.com/), [Momin Abbas](https://mominabbas.github.io/), Huzaifa Arif, [Tobias Park](https://kd2eom.github.io/), Xinyan Sun



Language: Python
API: Pytorch

# Instructions

The `pytorch/data` folder contains the datasets used for training (`MNIST`,`CIFAR-10`,`CIFAR-100`). Results of our experiments are located in the `pytorch/results` folder. 
To reproduce all of the experiments, simply download all the files in the pytorch folder and run `pytorch/main.py` as ``` python main.py ```.


Pytorch optimizers for all of the Hypergradient Descent algorithms are located in `pytorch/optimizers.py` if you want to utilize these algorithms for your own purposes or to run your own experiments. They are normal pytorch optimizers that can simply be plugged in and used in the same way as any normal pytorch optimizer. `pytorch/optimizer.py` includes:
  - `SGD` and `SGD-HD`
  - `SGDN` and `SGDN-HD`
  - `Adam` and `Adam-HD`


To select which models, dataset or optimizers you want to use to regenerate the results, just add that dataset in the respective files. ```pytorch/model.py``` contains model `classes`, ```pytorch/util.py``` has codes for dataloaders for datasets including `MNIST`, `CIFAR10` and `CIFAR100`. However, you can use your dataloader for your dataset(s). 


```pytroch/beta.py``` has the code that generates convergence trends for different `beta` values. 

# Reference
```
@article{baydin2017online,
  title={Online learning rate adaptation with hypergradient descent},
  author={Baydin, Atilim Gunes and Cornish, Robert and Rubio, David Martinez and Schmidt, Mark and Wood, Frank},
  journal={arXiv preprint arXiv:1703.04782},
  year={2017}
}
```


import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd


def save_plot (logs, model_name, folder ='results'):

    if (not os.path.exists(folder)):
        os.mkdir(folder)
    save_path = os.path.join (folder, model_name)
    if (not os.path.exists (save_path)):
        os.mkdir (save_path)
    
    keys = logs.keys()
    for k in keys:
        plt.plot(logs[k], label=k)
    
    plt.legend()
    plt.savefig (os.path.join( save_path, model_name+'.jpg'))


def save_csv (logs, model_name, folder ='results'):

    if (not os.path.exists(folder)):
        os.mkdir(folder)
    save_path = os.path.join (folder, model_name)
    if (not os.path.exists (save_path)):
        os.mkdir (save_path)
    
    df = pd.DataFrame(logs) 

    df.to_csv(os.path.join( save_path, model_name+'.csv'))
    

    

    


    

import pandas as pd
from utils.load_data import load_train
import numpy as np
import matplotlib.pyplot as plt
import random

def artificial_mixtures(dataset_index, x, y, depth, x_mixed=[], y_mixed=[], used_indexes=[]):
    if depth == 1:
        return x_mixed, y_mixed
    
    i1 = x[dataset_index]
    l1 = y.iloc[dataset_index].values

    random_mixture_index = random.randint(0,len(images)-1)
    while True:
        if random_mixture_index in used_indexes:
            random_mixture_index = random.randint(0,len(images)-1)
        else:
            
            i2 = x[random_mixture_index]
            l2 = y.iloc[random_mixture_index].values
            mixed_image = i1 + i2
            mixed_image = np.clip(mixed_image,0 , 255)
            mixed_label = l1 + l2
            return artificial_mixtures(dataset_index, x, y, depth-1, x_mixed+[mixed_image], y_mixed+[mixed_label], used_indexes+[random_mixture_index])
            




x, y  = load_train("./dataset")
x_normalized = np.array([i / 255.0 for i in x])

smiles = {"Smiles": y}
images = {"Image": x_normalized}

df_y = pd.DataFrame(smiles)
#df_x = pd.DataFrame(images)

oh = pd.get_dummies(df_y, prefix="")
#print(oh)
print(y[0])
#i1 = x_normalized[0]
#l1 = oh.iloc[0].values
#i2 = x_normalized[1]
#l2 = oh.iloc[1].values

x_mixed = []
y_mixed = []
for i in range(len(x_normalized)):

    x_mix, y_mix = artificial_mixtures(i, x_normalized, oh, 2)
    x_mixed += x_mix
    y_mixed += y_mix


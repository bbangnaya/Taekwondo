#1. module import
import os
import warnings
from tqdm import tqdm
from glob import glob
from PIL import Image
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


dfTrain = pd.read_csv("fashionmnist/fashion-mnist_train.csv")
dfTest = pd.read_csv("fashionmnist/fashion-mnist_test.csv")
print("Shape of Train Data: ", dfTrain.shape)
print("Shape of Test Data: ", dfTest.shape)


X_train = dfTrain.drop(["label"], axis=1)
Y_train = dfTrain.label
X_test = dfTest.drop(["label"], axis=1)
Y_test = dfTest.label


plt.figure(figsize=(20,5))

for i in range(10):
    plt.subplot(2, 5, i+1)
    img = dfTrain[dfTrain.label==i].iloc[0, 1:].values
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title("Class: " + str(i))
    plt.axis('off')
    
plt.show()
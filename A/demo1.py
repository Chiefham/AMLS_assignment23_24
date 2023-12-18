from data_loader import data_loader
import numpy as np
from ML_train_eva import ML
import os
from DL_train_eva import DL
import tensorflow as tf


parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = parent_folder + './Datasets/pneumoniamnist.npz'


x_train,x_val,x_test,y_train,y_val,y_test = data_loader(data_path)

model = DL(x_train,x_val,x_test,y_train,y_val,y_test)

model.LeNet5_train_eva_save()


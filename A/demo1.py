from data_loader import data_loader
import numpy as np
from ML_train import ML
import os

parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = parent_folder + './Datasets/pneumoniamnist.npz'


x_train,x_val,x_test,y_train,y_val,y_test = data_loader(data_path)

ML_model = ML(x_train,x_val,x_test,y_train,y_val,y_test)

ML_model.SVM()


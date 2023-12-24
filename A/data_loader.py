import numpy as np

# load the image data from .npz file
def data_loader(data_path):
    data = np.load(data_path)

    x_train = data['train_images']
    x_val = data['val_images']
    x_test = data['test_images']
    y_train = data['train_labels']
    y_val = data['val_labels']
    y_test = data['test_labels']
    return x_train,x_val,x_test,y_train,y_val,y_test




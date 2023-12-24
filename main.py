import numpy as np
from A.DL_train_eva import DL as A_DL
from A.ML_train_eva import ML as A_ML
from B.DL_train_eva import DL as B_DL



# load the data
def data_loader(data_path):
    data = np.load(data_path)

    x_train = data['train_images']
    x_val = data['val_images']
    x_test = data['test_images']
    y_train = data['train_labels']
    y_val = data['val_labels']
    y_test = data['test_labels']
    return x_train,x_val,x_test,y_train,y_val,y_test


def main():

    A_Data_path = './Datasets/pneumoniamnist.npz'
    B_Data_path = './Datasets/pathmnist.npz'
    A_x_train,A_x_val,A_x_test,A_y_train,A_y_val,A_y_test = data_loader(A_Data_path)
    B_x_train,B_x_val,B_x_test,B_y_train,B_y_val,B_y_test = data_loader(B_Data_path)

    ML_Model_A = A_ML(A_x_train, A_x_val, A_x_test, A_y_train, A_y_val, A_y_test)
    DL_Model_A = A_DL(A_x_train,A_x_val,A_x_test,A_y_train,A_y_val,A_y_test)
    DL_Model_B = B_DL(B_x_train,B_x_val,B_x_test,B_y_train,B_y_val,B_y_test)


    # Select the module you want to run
    # And comment out the ones you do not need to run

    ML_Model_A.SVM()
    # ML_Model_A.RF()
    # ML_Model_A.KNN()
    # DL_Model_A.MobileNetV2()
    # DL_Model_A.DNN()
    # DL_Model_A.LeNet5()
    #
    # DL_Model_B.MobileNetV2()
    # DL_Model_B.VGG19()
    # DL_Model_B.DenseNet()
    # DL_Model_B.ResNet152V2()




if __name__ == "__main__":
    main()

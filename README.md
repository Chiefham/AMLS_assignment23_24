# AMLS_assignment23_24


Description:

The project folder contains one main.py file 
that can be used by the user to quickly run the training and
test code for the models in the two tasks. README.md is 
used to introduce the files in the project. Folder A contains 
the training and testing code for the models needed to complete 
task A, and the well trained models. Folder B contains the training 
and testing code for the models needed to complete task B, and the 
well trained models. Datasets is an empty folder where you can paste 
the required datasets into for use.

----------------------------------------------------------
Packages:

numpy  
opencv-python  
keras  
scikit-learn  
tensorflow  
joblib




--------------------------------------------------------
Introduction for each file:


A/:

DNN.model: Saved DNN model .

LeNet5.model: Saved LeNet5 model.

MobileNetV2.model: Saved MobileNetV2 model.

Model_KNN: Saved KNN model.

Model_RF: Saved RF model.

Model_SVM: Saved SVM model.

DL_train_eva.py: It contains a class which has three main 
methods LeNet5(), MobileNetV2(), DNN() to call. Each method 
requires no input and has no return value. When a method is called, 
the corresponding deep learning model will be built and trained 
and evaluated, and the model will be saved. This file provides 
the code implementation of the deep learning model for task A.

ML_train_eva.py: It contains a class which has three 
main methods SVM(), RF(), KNN() to call. Each method requires 
no input and has no return value, when the method is called, 
the corresponding machine learning model will be built and 
trained and evaluated, and the model will be saved. 
This file provides the 
code implementation of the machine learning model for task A.

GridSearchCV_KNN.py: Use the GridSearchCV() function to 
tune the hyperparameters of the KNN model to find the combination of 
hyperparameters that will allow the model to perform best.

GridSearchCV_RF.py: Use the GridSearchCV() function to 
tune the hyperparameters of the RF model to find the combination of 
hyperparameters that will allow the model to perform best.

GridSearchCV_SVM.py: Use the GridSearchCV() function to 
tune the hyperparameters of the SVM model to find the combination of 
hyperparameters that will allow the model to perform best.

data_loader.py: Load the data.

--------------------------------
B/:

DenseNet201.model: Saved DenseNet201 model

MobileNetV2.model: Saved MobileNetV2 model

ResNet152V2.model: Saved ResNet152V2 model

VGG19.model: Saved VGG19 model

DL_train_eva.py: It contains a class which 
has four main methods to call inside the class, 
VGG19(), MobileNetV2(), DenseNet(), ResNet152V2(). 
Each method requires no inputs and has no return values, 
when the method is called, the corresponding deep learning 
model will be built and trained and evaluated, and the model 
will be saved. This file provides 
the code implementation of the deep learning model for Task B

data_loader.py: Load the data

-----------------
Datasets/: An empty folder to store the datasets

-------------------
main.py: A main function that can directly call the built classes 
and methods to train, evaluate and store the model.






import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib


class ML:
    def __init__(self,x_train,x_val,x_test,y_train,y_val,y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def SVM(self,kernel = 'rbf',C = 10,gamma='auto'):
        # Data Pre-Processing
        X = []
        for img in self.x_train:
            hist = cv2.calcHist([img],[0],None,[256],[0,255])
            X.append((hist/255).flatten())
        x_train = np.array(X)
        y_train = np.ravel(self.y_train)

        # Model Training
        clf = SVC(kernel = kernel,C = C,gamma = gamma)
        clf.fit(x_train,y_train)
        joblib.dump(filename="Model_SVM",value=clf)

    def RF(self,n_estimators):
        clf = RandomForestClassifier(n_estimators = n_estimators,random_state=7)
        clf.fit(self.x_train,self.y_train)
        joblib.dump(filename="Model_RF",value=clf)

    def KNN(self,n_neighbors):
        clf = KNeighborsClassifier(n_neighbors =n_neighbors)
        clf.fit(self.x_train,self.y_train)
        joblib.dump(filename="Model_KNN",value=clf)



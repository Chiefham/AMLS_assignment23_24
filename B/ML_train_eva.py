import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score


class ML:
    def __init__(self,x_train,x_val,x_test,y_train,y_val,y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def SVM(self,kernel = 'rbf',C = 1.5,gamma=60):
        # Data Pre-Processing
        x_train,y_train = self.transformer(self.x_train,self.y_train)
        x_test,y_test = self.transformer(self.x_test,self.y_test)


        # Model Training
        clf = SVC(kernel = kernel,C = C,gamma = gamma)
        clf.fit(x_train,y_train)


        # Model Evaluation
        y_predict = clf.predict(x_test)
        ACC = accuracy_score(y_test,y_predict)
        PRE = precision_score(y_test,y_predict)
        RECALL = recall_score(y_test,y_predict)
        F1 = f1_score(y_test,y_predict)
        KAPPA = cohen_kappa_score(y_test,y_predict)

        print("Accuracy:",ACC)
        print("Precision:",PRE)
        print("Recall:",RECALL)
        print("F1 Score:",F1)
        print("Cohen’s Kappa Coefficient:",KAPPA)


        # Model Save
        joblib.dump(filename="Model_SVM", value=clf)

    def RF(self,n_estimators=81,max_features=8):
        # Data Pre-Processing
        x_train, y_train = self.transformer(self.x_train, self.y_train)
        x_test, y_test = self.transformer(self.x_test, self.y_test)

        # Model Training
        clf = RandomForestClassifier(n_estimators = n_estimators,max_features=max_features,random_state=7)
        clf.fit(x_train,y_train)

        # Model Evaluation
        y_predict = clf.predict(x_test)
        ACC = accuracy_score(y_test, y_predict)
        PRE = precision_score(y_test, y_predict)
        RECALL = recall_score(y_test, y_predict)
        F1 = f1_score(y_test, y_predict)
        KAPPA = cohen_kappa_score(y_test, y_predict)

        print("Accuracy:", ACC)
        print("Precision:", PRE)
        print("Recall:", RECALL)
        print("F1 Score:", F1)
        print("Cohen’s Kappa Coefficient:", KAPPA)

        # Model Save
        joblib.dump(filename="Model_RF",value=clf)

    def KNN(self,n_neighbors=31):
        # Data Pre-Processing
        x_train, y_train = self.transformer(self.x_train, self.y_train)
        x_test, y_test = self.transformer(self.x_test, self.y_test)

        # Model Training
        clf = KNeighborsClassifier(n_neighbors =n_neighbors)
        clf.fit(x_train, y_train)

        # Model Evaluation
        y_predict = clf.predict(x_test)
        ACC = accuracy_score(y_test, y_predict)
        PRE = precision_score(y_test, y_predict)
        RECALL = recall_score(y_test, y_predict)
        F1 = f1_score(y_test, y_predict)
        KAPPA = cohen_kappa_score(y_test, y_predict)

        print("Accuracy:", ACC)
        print("Precision:", PRE)
        print("Recall:", RECALL)
        print("F1 Score:", F1)
        print("Cohen’s Kappa Coefficient:", KAPPA)

        # Model Save
        joblib.dump(filename="Model_KNN",value=clf)

    def transformer(self,x,y):
        xx = []
        for img in x:
            img = img.reshape(28*28,3)
            xx.append(img)
        X = np.array(xx)
        Y = np.ravel(y)

        return X,Y


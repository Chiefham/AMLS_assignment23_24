from data_loader import data_loader
import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()

parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = parent_folder + './Datasets/pathmnist.npz'
x_train,x_val,x_test,y_train,y_val,y_test = data_loader(data_path)
#
# SVM_parameters = {
#     'kernel':('linear','rbf'),
#     'C':[1,0.8,0.5,1.5,0.1,0.01],
#     'gamma':[20,25,30,35,40,45,50,60,70]
# }

SVM_parameters = {
    'kernel':('rbf','linear'),
    'C':[0.8],
    'gamma':[60]
}


X = []
for img in x_train:
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    X.append((hist / 255).flatten())
x_train = np.array(X)
y_train = np.ravel(y_train)

svr = svm.SVC()
clf = GridSearchCV(svr, SVM_parameters, scoring='f1')
clf.fit(x_train,y_train)
print(clf.cv_results_)

print("Best parameters:")
print(clf.best_params_)
print("score:")
print(clf.best_score_)

end_time = time.time()

print("Running Time:")
print((end_time-start_time))


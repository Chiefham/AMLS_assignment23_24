from data_loader import data_loader
import os
import numpy as np
import cv2
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = parent_folder + './Datasets/pneumoniamnist.npz'
x_train,x_val,x_test,y_train,y_val,y_test = data_loader(data_path)

RF_parameters = {
    "n_estimators":range(1,101,10),
    "max_features":range(1,11,1)
}

X = []
for img in x_train:
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    X.append((hist / 255).flatten())
x_train = np.array(X)
y_train = np.ravel(y_train)

rf = RandomForestClassifier()
clf = GridSearchCV(rf, RF_parameters, scoring='f1')
clf.fit(x_train,y_train)

print(clf.cv_results_)

print("Best parameters:")
print(clf.best_params_)
print("score:")
print(clf.best_score_)

end_time = time.time()

print("Running Time:")
print((end_time-start_time))


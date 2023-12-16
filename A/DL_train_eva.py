import joblib
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
from keras.utils import to_categorical
from data_loader import data_loader
import tensorflow as tf
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report, cohen_kappa_score
import numpy as np
from keras.callbacks import EarlyStopping
from keras.applications.resnet import ResNet50


class LeNet5:
    def __init__(self):
        self.model = Sequential()


    def data_transformer(self,x,y):
        X = x.reshape(-1,28,28,1)
        X = X/255
        Y = to_categorical(y,num_classes=2)

        return X,Y

    def train_eva_save(self,x_train,y_train,x_val,y_val,x_test,y_test):
        # Model Structure
        self.model.add(Conv2D(input_shape=(28,28,1),
                              filters=6,
                              kernel_size=(5,5),
                              strides=1,
                              activation=keras.activations.tanh
                              )
        )

        self.model.add(AveragePooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(filters=6,
                              kernel_size=(5, 5),
                              strides=1,
                              activation=tf.keras.activations.tanh))

        self.model.add(AveragePooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))




        # Model Training
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer='RMSprop',
            metrics = ['accuracy'],
        )
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        self.model.fit(x=x_train,y=y_train,batch_size=32,epochs=50,verbose=1,
                       callbacks=[early_stopping],
                       )

        # Model Evaluation
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes,digits=6)
        kappa = cohen_kappa_score(y_true_classes, y_pred_classes)

        print(report)

        print(kappa)


        # Model Save

        self.model.save("LeNet5.model")
































import cv2
import keras
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense,Dropout
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.resnet_v2 import ResNet152V2





class DL:
    def __init__(self,x_train,x_val,x_test,y_train,y_val,y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def data_transformer(self,x,y):
        X = []
        for img in x:
            img = cv2.resize(img,(32,32))
            X.append(img)
        X = np.array(X)
        X = X/255
        Y = to_categorical(y,num_classes=9)

        return X,Y

    def model_eva(self,model,x_test,y_test):
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes, digits=6)
        print(report)

    def top_layers(self,dl_model):
        for layer in dl_model.layers[:-1]:
            layer.trainable = False
        top_model = Sequential()  # construct the top network
        top_model.add(Flatten(input_shape=dl_model.output_shape[1:]))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(9, activation='softmax'))  # binary classfication
        model = Model(
            inputs=dl_model.input,
            outputs=top_model(dl_model.output)
        )

        return model

    def VGG19(self):
        # Data Pre_Processing
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)


        model_vgg19 = VGG19(
            weights="imagenet", include_top=False,
            input_shape=(32, 32, 3)
        )
        for layer in model_vgg19.layers[:-1]:
            layer.trainable = False
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model_vgg19.output_shape[1:]))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(9, activation='softmax'))
        model = Model(
            inputs=model_vgg19.input,
            outputs=top_model(model_vgg19.output)
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        model.fit(x_train, y_train, batch_size=64,
                  epochs=50,
                  callbacks=[callback],
                  validation_data=(x_val,y_val)
                  )

        # Model Evaluation
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes, digits=6)
        print(report)

        # Model Save
        model.save("VGG19.model")

    def MobileNetV2(self):
        # Data Pre_Processing
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)

        model_MobileNetV2 = MobileNetV2(
            weights="imagenet", include_top=False,
            input_shape=(32, 32, 3)
        )

        model = self.top_layers(model_MobileNetV2)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        model.fit(x_train, y_train, batch_size=64,
                  epochs=50,
                  callbacks=[callback],
                  validation_data=(x_val, y_val)
                  )
        self.model_eva(model,x_test,y_test)

        # Model Save
        model.save("MobileNetV2.model")

    def DenseNet(self):
        # Data Pre_Processing
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)

        model_DenseNet201 = DenseNet201(
            weights="imagenet", include_top=False,
            input_shape=(32, 32, 3)
        )

        model = self.top_layers(model_DenseNet201)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        model.fit(x_train, y_train, batch_size=64,
                  epochs=50,
                  callbacks=[callback],
                  validation_data=(x_val, y_val)
                  )
        self.model_eva(model, x_test, y_test)

        # Model Save
        model.save("DenseNet.model")


    def ResNet152V2(self):
        # Data Pre_Processing
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)

        model_ResNet152V2 = ResNet152V2(
            weights="imagenet", include_top=False,
            input_shape=(32, 32, 3)
        )

        model = self.top_layers(model_ResNet152V2)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        model.fit(x_train, y_train, batch_size=64,
                  epochs=50,
                  callbacks=[callback],
                  validation_data=(x_val, y_val)
                  )
        self.model_eva(model, x_test, y_test)

        # Model Save
        model.save("ResNet152V2.model")














































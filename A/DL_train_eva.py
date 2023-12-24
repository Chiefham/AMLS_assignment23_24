import keras
from keras.layers import AveragePooling2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
from keras.callbacks import EarlyStopping
from keras import layers
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense





class DL:
    def __init__(self,x_train,x_val,x_test,y_train,y_val,y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test




    def data_transformer(self,x,y):
        # data preprocessing for deep learning in task A
        X = x.reshape(-1,28,28,1)
        X = X/255
        Y = to_categorical(y,num_classes=2)

        return X,Y



    def LeNet5(self):
        # load the data
        x_train,y_train = self.data_transformer(self.x_train,self.y_train)
        x_val,y_val = self.data_transformer(self.x_val,self.y_val)
        x_test,y_test = self.data_transformer(self.x_test,self.y_test)

        # Model Structure
        model = Sequential()
        model.add(Conv2D(input_shape=(28,28,1),
                         filters=6,
                         kernel_size=(5,5),
                         strides=1,
                         activation='relu'
                         )
                  )
        model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=16,
                         kernel_size=(5, 5),
                         strides=1,
                         activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # model compilation and training
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer='RMSprop',
            metrics = ['accuracy'],
        )

        # apply eary stopping technology
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(x_train,y_train,batch_size=32,epochs=50,verbose=1,
                  callbacks=[early_stopping],
                  validation_data=(x_val,y_val),
                  )

        # Model Evaluation
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes,digits=6)
        print(report)

        # Model Save
        model.save("LeNet5.model")

    def MobileNetV2(self):
        # Data Pre_Processing
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)

        # Model Structure
        def conv_block(inputs, filters, kernel_size, strides):
            x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=1):
            x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides)(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(pointwise_conv_filters, (1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        input_shape = (28, 28, 1)
        num_classes = 2
        inputs = Input(shape=input_shape)
        x = conv_block(inputs, 32, (3, 3), strides=(2, 2))
        for _ in range(13):
            x = depthwise_conv_block(x, 32)
        x = depthwise_conv_block(x, 64, strides=(2, 2))
        for _ in range(13):
            x = depthwise_conv_block(x, 64)
        x = depthwise_conv_block(x, 128, strides=(2, 2))
        for _ in range(13):
            x = depthwise_conv_block(x, 128)
        x = depthwise_conv_block(x, 256, strides=(2, 2))
        for _ in range(5):
            x = depthwise_conv_block(x, 256)
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='sigmoid')(x)
        model = Model(inputs, x, name='MobileNetV2')

        # Model Training
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val),
                  callbacks=[early_stopping],
                  )

        # Model Evaluation
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes, digits=6)
        print(report)

        # Model Save
        model.save("MobileNetV2.model")

    def DNN(self):
        # load data
        x_train, y_train = self.data_transformer(self.x_train, self.y_train)
        x_val, y_val = self.data_transformer(self.x_val, self.y_val)
        x_test, y_test = self.data_transformer(self.x_test, self.y_test)

        # Model Structure
        model = models.Sequential()
        model.add(Flatten(input_shape=(28,28)))
        model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        model.add(layers.Dense(2, activation='softmax'))

        # Model Training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(x_train,
                  y_train,
                  epochs=50,
                  batch_size=32,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping],
                  )

        # Model Evaluation
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes, digits=6)
        print(report)

        # Model Save
        model.save("DNN.model")














































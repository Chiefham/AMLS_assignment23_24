import joblib
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
from keras.utils import to_categorical


class DL:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def VGG19(self,learning_rate=0.001,epochs=100,batch_size=32):
        # Data Pre-Processing
        x_train,y_train = self.transformer(self.x_train,self.y_train)
        x_val,y_val = self.transformer(self.x_val,self.y_val)
        x_test,y_test = self.transformer(self.x_test,self.y_test)


        # Model Training
        Model = VGG19(weights="imagenet", include_top=False,input_shape=(28,28,1))
        for layer in Model.layers[:-1]:
            layer.trainable = False
        top_model = Sequential()  # construct the top network
        top_model.add(Flatten(input_shape=Model.output_shape[1:]))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(2, activation='softmax'))  # binary classfication

        model = Model(
            inputs=Model.input,
            outputs=top_model(Model.output)
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy',keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.F1Score()]
        )
        callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)

        history = model.fit(x_train,y_train,
                            batch_size = batch_size,
                            epochs = epochs,validation_data = (x_val,y_val),
                            callbacks = callback)


        # Model Evaluation
        print(history.history)





        # Model Save
        model.save('./VGG19.model')

        pass

    def DenseNet201(self):

        pass

    def InceptionV3(self):

        pass

    def transformer(self,x,y):
        X = x.reshape(-1,28,28,1)
        X = X/255
        Y = to_categorical(y,num_classes=2)

        return X,Y

class LeNet5:
    def __init__(self):
        self.model = Sequential()



























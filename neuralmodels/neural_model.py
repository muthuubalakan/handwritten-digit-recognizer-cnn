#! /usr/bin/env python3
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K

import os


# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

n_classes = y_test.shape[1]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation="relu",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation="relu",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(lr=0.01,
                                           decay=1e-6,
                                           momentum=0.9,
                                           nesterov=True), metrics=["accuracy"])
    return model

# call
model = CNN_model()

# train
model.fit(x_train, y_train, batch_size=128,
          epochs=10, verbose=1,
          validation_data=(x_test, y_test))

progress = model.evaluate(x_test, y_test, verbose=0)
print("Loss:{}\nAccuracy: {}\n".format(progress[0], progress[1]))

# save model
if not os.path.isfile("CNNDigit.h5"):
    print("Saving the model...")
    model.save("CNNDigit.h5")

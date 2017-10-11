import  keras
import random
import numpy as np
from keras.layers import Dense,Activation
from keras.activations import sigmoid,softmax
from keras.models import Sequential
from keras.optimizers import SGD,adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils


def one_hot_encode_object_array(arr):
    '''''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


def creat_data(min,max,num):
    x = np.zeros(num)
    y = np.zeros(num)
    lables = np.zeros(num)
    for i in range(1, num):
        x[i] = random.uniform(min, max)
        y[i] = random.uniform(min, max)
        if y[i]>x[i]*x[i]:
            lables[i]=1
        else:
            lables[i] = 0
    x_return=np.c_[x,y]
    return x_return,lables
x_train,y_train=creat_data(0,10,10000)
x_test,y_test=creat_data(0,10,500)
lables_train=one_hot_encode_object_array(y_train)
lables_test=one_hot_encode_object_array(y_test)
model=Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
score=model.fit(x_train,lables_train,batch_size=128)
loss, accuracy =model.evaluate(x_test,lables_test,batch_size=10)
print("Accuracy = {:.2f}".format(accuracy))

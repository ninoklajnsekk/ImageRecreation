from keras.datasets import cifar10
import numpy as np

def filter_dataset(label_number,amount,split):
    class_i = 0
    opposite_i = 0
    new_xtrain = []
    new_ytrain = []
    new_xtest = []
    new_ytest = []

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    for idx, x in enumerate(x_train):
        if y_train[idx][0] == label_number & class_i<amount:
            new_xtrain.append(x_train[idx])
            y_train[idx][0] = 1
            new_ytrain.append(y_train[idx])
            class_i += 1;
        else:
            if opposite_i<amount:
                new_xtrain.append(x_train[idx])
                y_train[idx][0] = 0
                new_ytrain.append(y_train[idx])
                opposite_i += 1

    opposite_i = 0
    class_i = 0

    for idx, x in enumerate(x_test):
        if y_test[idx][0] == label_number & class_i<amount*split:
            new_xtest.append(x_test[idx])
            y_test[idx][0] = 1
            new_ytest.append(y_test[idx])
            class_i += 1
        else:
            if opposite_i<amount*split:
                new_xtest.append(x_test[idx])
                y_test[idx][0] = 0
                new_ytest.append(y_test[idx])
                opposite_i +=1


    return (np.array(new_xtrain), np.array(new_ytrain)), (np.array(new_xtest),np.array(new_ytest))

def keras10_dataset():
    return cifar10.load_data()

def dogcat_dataset():
    return "wat"

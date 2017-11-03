import sys
sys.path.append('libsvm-3.22/python/')
import math
import random
import numpy as np
from svmutil import *
import matplotlib.pyplot as plt

def get_parameter_range(initial_val, val_range):
    parameter_values = np.zeros((val_range),np.float)
    for i in range(0,13):
        parameter_values[i] = initial_val
        initial_val = initial_val*2
    return parameter_values

def train_and_test_SVM(y_train, x_train, y_test, x_test, svm_setting):
    problem = svm_problem(y_train, x_train)
    parameter = svm_parameter(svm_setting)
    model = svm_train(problem, parameter)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
    return p_acc[0]

def linear_SVM(y_train, x_train, y_test, x_test, C):
    SVM_accuracy = np.zeros((C.shape[0]),np.float)
    for i in range(0,C.shape[0]):
        svm_setting = '-t 0 -c '+ str(C[i])
        accuracy = train_and_test_SVM(y_train, x_train, y_test, x_test, svm_setting)
        SVM_accuracy[i] = accuracy
    fig = plt.figure()
    plt.title('Linear SVM: Accuracy vs log(C)')
    plt.plot(np.log(C)/np.log(2),SVM_accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('log(C)')
    fig.savefig('results/linearSVM.png')
    plt.show()

def RBF_CV_SVM(y_train, x_train, y_test, x_test, C, Alpha, cv_size):
    train_size = len(y_train)

    sample_index = random.sample(range(0, train_size), train_size/2)
    x_train_half = np.array(x_train)[sample_index]
    y_train_half = np.array(y_train)[sample_index]

    x_cv_train = []
    y_cv_train = []
    x_cv_test = []
    y_cv_test = []
    SVM_accuracy = np.zeros((C.shape[0],Alpha.shape[0]),np.float)

    for i in range(0,cv_size):
        range1_end = (i*train_size/(2*cv_size))
        range2_start = ((i+1)*train_size/(2*cv_size))
        x_cv_train.append(x_train_half[np.r_[0:range1_end,range2_start:(train_size/2)]])
        y_cv_train.append(y_train_half[np.r_[0:range1_end,range2_start:(train_size/2)]])
        x_cv_test.append(x_train_half[range1_end:range2_start])
        y_cv_test.append(y_train_half[range1_end:range2_start])

    for i in range(0,C.shape[0]):
        for j in range(0,Alpha.shape[0]):
            for k in range(0,cv_size):
                svm_setting = '-t 2 -c '+ str(C[i]) + ' -g ' + str(Alpha[j])
                accuracy = train_and_test_SVM(y_cv_train[k], x_cv_train[k], y_cv_test[k], x_cv_test[k], svm_setting)
                SVM_accuracy[i,j] = SVM_accuracy[i,j] + accuracy
            SVM_accuracy[i,j] = SVM_accuracy[i,j] / cv_size

    print SVM_accuracy
    index_x,index_y = np.where(SVM_accuracy == SVM_accuracy.max())
    print index_x,index_y
    svm_setting = '-t 2 -c '+ str(C[index_x[0]]) + ' -g ' + str(Alpha[index_y[0]])
    accuracy = train_and_test_SVM(y_train, x_train, y_test, x_test, svm_setting)
    print accuracy

if __name__ == "__main__":
    initial_val = 1.0/(math.pow(2,4))
    cv_size = 5
    C = get_parameter_range(initial_val, 13)
    Alpha = get_parameter_range(initial_val, 13)
    y_train, x_train = svm_read_problem('ncrna_s.train.txt')
    y_test, x_test = svm_read_problem('ncrna_s.train.txt')
    linear_SVM(y_train, x_train, y_test, x_test, C)
    RBF_CV_SVM(y_train, x_train, y_test, x_test, C, Alpha, cv_size)

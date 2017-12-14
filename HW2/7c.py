import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def loss_error(X,omega,Y):
    h = 1 / (1 + np.exp(-np.dot(X,omega)))
    return -np.sum(Y * np.log(h) + (1 - Y) * np.log(1-h))

def classification_error(X,omega,Y):
    misclassify = 0
    re = np.dot(X,omega)
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != 0):
            misclassify += 1
    return misclassify / len(re)

def gradientDescent(X, Y, omega, alpha, iters, lmd):
    while iters != 0:
        h = 1 / (1 + np.exp(-np.dot(X,omega)))
        error = Y - h
        error = np.dot(X.T, error)
        
        omega = omega + alpha * (error + lmd * omega)
            
        iters = iters - 1
    return omega

def k_fold(X,Y,k,lmd):
    length = X.shape[0] // k
    k_error = 0
    for i in range(k):
        theta_init = np.zeros((X.shape[1],1))
        k_x_test = X[i*length:(i+1)*length-1, :]
        k_y_test = Y[i*length:(i+1)*length-1, :]
        if i == 0:
            k_x_train = X[(i+1)*length-1:, :]
            k_y_train = Y[(i+1)*length-1:, :]
        elif i == k-1:
            k_x_train = X[0:i*length, :]
            k_y_train = Y[0:i*length, :]
        else:
            k_x_train = np.zeros((X.shape[0] - length, X.shape[1]))
            k_y_train = np.zeros((Y.shape[0] - length, Y.shape[1]))
            k_x_train[0:i*length, :] = X[0:i*length, :] 
            k_x_train[i*length:,:] = X[(i+1)*length:,:]
            k_y_train[0:i*length, :] = Y[0:i*length, :] 
            k_y_train[i*length:,:] = Y[(i+1)*length:,:]
        
        theta_res = gradientDescent(k_x_train, k_y_train, theta_init, 0.01, 1000, lmd)
        error = classification_error(k_x_test,theta_res,k_y_test)
        k_error = k_error + error
        
    return k_error / k

if __name__ == '__main__':
    data = sio.loadmat('data2.mat')
    X_train = data['X_trn']
    Y_train = data['Y_trn']
    X_test = data['X_tst']
    Y_test = data['Y_tst']

    X_train = np.insert(X_train, 0, 1.0, axis=1)
    X_test = np.insert(X_test, 0, 1.0, axis=1)

    # k-fold CV to choose lamda
    error = []
    lmd_candidates = [0.00001, 0.001, 0.01, 0.1]
    for l in lmd_candidates:
        error.append(k_fold(X_train,Y_train,3,l))
    lmd = lmd_candidates[error.index(min(error))]

    alpha = 0.01
    iters = 1000
    omega = np.zeros((X_train.shape[1],1))

    omega = gradientDescent(X_train, Y_train, omega, alpha, iters, lmd)

    classify_error_training= classification_error(X_train,omega,Y_train)
    classify_error_testing = classification_error(X_test,omega,Y_test)

    print("Omega = ", omega.T)
    print("Classification error on training data = ", classify_error_training)
    print("Classification error on testing data = ", classify_error_testing)

    x = np.linspace(min(X_train[:,1]), max(X_train[:,1]), 100)
    f = (-omega[0] - omega[1] * x) / omega[2]

    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(len(Y_train)):
        if Y_train[i] != 0:
            break
    ax.scatter(X_train[0:i-1,1], X_train[0:i-1,2], c='red', marker = '+', label='training data class 0')
    ax.scatter(X_train[i:,1], X_train[i:,2], c='blue', marker = '+', label='training data class 1')
    ax.plot(x, f, 'g', label = 'decision boundary')
    for j in range(len(Y_test)):
        if Y_test[j] != 0:
            break
    ax.scatter(X_test[0:j-1,1], X_test[0:j-1,2], c='red', marker = '^', label='testing data class 0')
    ax.scatter(X_test[j:,1], X_test[j:,2], c='blue', marker = '^', label='testing data class 1')
    ax.legend(loc=2)
    plt.show()
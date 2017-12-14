import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def phi_nonlinear(X, n):
    length = len(X)
    X_res = np.ones((length, n+1))
    for i in range(1,n+1):
        X_res[:,i] = (X**i).T
    return X_res

def closed_form(X,Y):
    part1 = np.linalg.inv(np.dot(X.T, X))
    part2 = np.dot(X.T, Y)
    return np.dot(part1, part2)

def cal_error(X,theta,Y):
    return np.sum(np.power((Y-np.dot(X,theta)),2)) / len(Y)

def Stochastic_Descent(X, Y, theta, alpha, iters, m):
    cur_iter = 0
    temp = np.zeros(theta.shape)
    
    while cur_iter != iters:
        sample = random.sample(range(0,120), m)
        batch_X = np.zeros((m,X.shape[1]))
        batch_Y = np.zeros((m,1))
        k = 0
        for i in sample:
            batch_X[k,:] = X[i,:]
            batch_Y[k,:] = Y[i,:]
            k = k + 1
        error = np.dot(batch_X, theta) - batch_Y
        error = np.dot(batch_X.T, error)
            
        theta = theta - (alpha / m) * error
        cur_iter = cur_iter + 1
    return theta

if __name__ == '__main__':
    data = sio.loadmat('dataset1.mat')
    X_train = data['X_trn']
    Y_train = data['Y_trn']
    X_test = data['X_tst']
    Y_test = data['Y_tst']
    n = [2,3,5]
    print('Solution for the implementation of closed-form')
    for i in n:
        print('n =', i)
        X_train_new = phi_nonlinear(X_train, i)
        theta_res = closed_form(X_train_new, Y_train)
        print('Theta =', theta_res.T)
        train_error = cal_error(X_train_new,theta_res,Y_train)
        print('Training error =',train_error)
        X_test_new = phi_nonlinear(X_test, i)
        test_error = cal_error(X_test_new,theta_res,Y_test)
        print('Testing error =', test_error)
        print('------------------------------------------')


    print('Solution for the implementation of stochastic gradient decent')
    for i in n:
    	print('n =', i)
    	theta_init = np.zeros((i+1,1))
    	X_train_new = phi_nonlinear(X_train, i)
    	theta_res = Stochastic_Descent(X_train_new, Y_train, theta_init, 0.00001, 1000000, 30)
    	print('Theta =', theta_res.T)
    	train_error = cal_error(X_train_new,theta_res,Y_train)
    	print('Training error =',train_error)
    	X_test_new = phi_nonlinear(X_test, i)
    	test_error = cal_error(X_test_new,theta_res,Y_test)
    	print('Testing error =', test_error)
    	print('------------------------------------------')

    print('The bigger the size of the mini-batch, the smaller the testing error but the slower of the speed.')


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

def closed_form(X,Y,lmd):
    part1 = np.linalg.inv(np.dot(X.T, X) + lmd * np.identity(X.shape[1]))
    part2 = np.dot(X.T, Y)   
    return np.dot(part1, part2)

def cal_error(X,theta,Y):
    return np.sum(np.power((Y-np.dot(X,theta)),2)) / len(Y)

def Stochastic_Descent(X, Y, theta, alpha, iters, m, lmd):
    cur_iter = 0
    temp = np.zeros(theta.shape)
    
    while cur_iter != iters:
        sample = random.sample(range(0,X.shape[0]), m)
        batch_X = np.zeros((m,X.shape[1]))
        batch_Y = np.zeros((m,1))
        k = 0
        for i in sample:
            batch_X[k,:] = X[i,:]
            batch_Y[k,:] = Y[i,:]
            k = k + 1
        error = np.dot(batch_X, theta) - batch_Y
        error = np.dot(batch_X.T, error)
            
        theta = theta - (alpha / m) * (error + lmd * theta)
        cur_iter = cur_iter + 1
    return theta

def k_ford_closed(X,Y,k,lmd):
    length = X.shape[0] // k
    k_error = 0
    for i in range(k):
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
        
        theta_res = closed_form(k_x_train, k_y_train, lmd)
        error = cal_error(k_x_test,theta_res,k_y_test)
        k_error = k_error + error
        
    return k_error / k

def k_ford_stochastic(X,Y,k,lmd):
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
        
        if k_x_train.shape[1] > 5:
        	theta_res = Stochastic_Descent(k_x_train, k_y_train, theta_init, 0.00000001, 10000, 30, lmd)
        else:
        	theta_res = Stochastic_Descent(k_x_train, k_y_train, theta_init, 0.00001, 10000, 30, lmd)
        error = cal_error(k_x_test,theta_res,k_y_test)
        k_error = k_error + error
        
    return k_error / k


if __name__ == '__main__':
	data = sio.loadmat('dataset2.mat')
	X_train = data['X_trn']
	Y_train = data['Y_trn']
	X_test = data['X_tst']
	Y_test = data['Y_tst']
	lmd_candidates = [0.01, 0.1, 1, 10, 100]
	print('Given lambda candidates list as', lmd_candidates)
	print('                                               ')
	N = [2,3,5]
	K = [2,10,100]

	print('Solution for the implementation of closed-form')
	for n in N:
	    for k in K:
	        X_train_new = phi_nonlinear(X_train, n)
	        error = []
	        for i in lmd_candidates:
	            error.append(k_ford_closed(X_train_new, Y_train, k, i))
	    
	        opt_lmd = lmd_candidates[error.index(min(error))]
	        print('n =', n)
	        print('k =', k)
	        print('The best lambda =', opt_lmd)
	        theta_init = np.zeros((X_train_new.shape[1],1))
	        theta_res = closed_form(X_train_new, Y_train, opt_lmd)
	        print('Theta =', theta_res)
	        train_error = cal_error(X_train_new,theta_res,Y_train)
	        print('Training error =', train_error)
	        X_test_new = phi_nonlinear(X_test, n)
	        test_error = cal_error(X_test_new,theta_res,Y_test)
	        print('Testing error =', test_error)
	        print('------------------------------------------')

	print('Solution for the implementation of stochastic gradient decent')
	for n in N:
	    for k in K:
	        X_train_new = phi_nonlinear(X_train, n)
	        error = []
	        for i in lmd_candidates:
	            error.append(k_ford_stochastic(X_train_new, Y_train, k, i))
	    
	        opt_lmd = lmd_candidates[error.index(min(error))]
	        print('n =', n)
	        print('k =', k)
	        print('The best lambda =', opt_lmd)
	        theta_init = np.zeros((X_train_new.shape[1],1))
	        if n == 5:
	        	theta_res = Stochastic_Descent(X_train_new, Y_train, theta_init, 0.00000001, 10000, 30, opt_lmd)
	        else:
	        	theta_res = Stochastic_Descent(X_train_new, Y_train, theta_init, 0.00001, 10000, 30, opt_lmd)
	        print('Theta =', theta_res)
	        train_error = cal_error(X_train_new,theta_res,Y_train)
	        print('Training error =', train_error)
	        X_test_new = phi_nonlinear(X_test, n)
	        test_error = cal_error(X_test_new,theta_res,Y_test)
	        print('Testing error =', test_error)
	        print('------------------------------------------')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc

#------------------PCA------------------------
def reg_PCA(Y_train, Y_test, d):
    row, col = Y_train.shape
    mu_bar = Y_train.mean(1).reshape(row,1)
    Y_train_temp = Y_train - mu_bar
    row, col = Y_test.shape
    mu_bar = Y_test.mean(1).reshape(row,1)
    Y_test_temp = Y_test - mu_bar
    U,S,V = np.linalg.svd(Y_train_temp)
    return np.dot(U[:,0:d].T, Y_train_temp), np.dot(U[:,0:d].T, Y_test_temp)
#----------------------------------------------

#-------------Logistic regression--------------
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
    m = len(Y)
    while iters != 0:
        h = 1 / (1 + np.exp(-np.dot(X,omega)))
        error = Y - h
        error = np.dot(X.T, error)
        omega = omega + alpha * (error + lmd * omega)
        iters = iters - 1     
    return omega

if __name__ == '__main__':
	# load data
	data = sio.loadmat('ExtYaleB10.mat')
	train_samples = data['train']
	test_samples = data['test']

	# reshape data
	X_train = np.column_stack(train_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(50))
	I = np.identity(10)
	Y_train = np.column_stack(I[:,i] for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	Y_test = np.column_stack(I[:,i] for i in range(10) for j in range(14))
	X_train = X_train / 255
	X_test = X_test / 255

	# apply PCA
	X_train_100, X_test_100 = reg_PCA(X_train, X_test, 100)
	X_train = X_train_100.T
	X_test = X_test_100.T

	# train 1 vs all classifier on 100 dimensional data
	decision_boundaries = []
	alpha = 0.0001
	iters = 9400
	lmbda = 0.0001
	for i in range(10):
		Y_train = np.zeros([X_train.shape[0],1])
		Y_test = np.zeros([X_test.shape[0], 1])
		omega = np.zeros((X_train.shape[1],1))
		Y_train[i*50:(i+1)*50,0] = 1
		Y_test[i*14:(i+1)*14,0] = 1
		omega = gradientDescent(X_train, Y_train, omega, alpha, iters, lmbda)
		train_error = classification_error(X_train,omega,Y_train)
		test_error = classification_error(X_test,omega,Y_test)
		decision_boundaries.append([omega, train_error, test_error])
		print('Finish ',i, ' people')
	dec = np.column_stack(decision_boundaries[i][0] for i in range(10))

	# test the classifier on testing data
	res = np.dot(X_test/255, dec)
	error = 0
	label = 0
	for i in range(140):
		if i!=0 and i%14 == 0:
			label += 1
		if np.argmax(res[i,:]) != label:
			error += 1
	print('The classification error on test data is ', error / 140)
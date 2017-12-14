import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio


#----------------logistic regression-------------------
def loss_error(X,omega,Y):
    h = 1 / (1 + np.exp(-np.dot(X,omega)))
    return -np.sum(Y * np.log(h) + (1 - Y) * np.log(1-h))

def classification_error(X,omega,Y):
    misclassify = 0
    re = np.dot(X,omega)
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != 0):
            misclassify += 1
    print(misclassify)
    print(len(re))
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
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	X_train = X_train.T
	X_test = X_test.T

	# train 1 vs all decision boundaries 
	decision_boundaries = []
	alpha = 0.0001
	iters = 1000
	lmbda = 0.001
	for i in range(10):
		Y_train = np.zeros([X_train.shape[0],1])
		Y_test = np.zeros([X_test.shape[0], 1])
		omega = np.zeros((X_train.shape[1],1))
		Y_train[i*50:(i+1)*50,0] = 1
		Y_test[i*14:(i+1)*14,0] = 1
		omega = gradientDescent(X_train/255, Y_train, omega, alpha, iters, lmbda)
		train_error = classification_error(X_train/255,omega,Y_train)
		test_error = classification_error(X_test/255,omega,Y_test)
		decision_boundaries.append([omega, train_error, test_error])
		print('Finish ',i, ' people')
	# testing 
	dec = np.column_stack(decision_boundaries[i][0] for i in range(10))
	res = np.dot(X_test/255, dec)
	error = 0
	label = 0
	for i in range(140):
		if i!=0 and i%14 == 0:
			label += 1
		if np.argmax(res[i,:]) != label:
			error += 1
	print('The classification error on test data is ', error / 140)
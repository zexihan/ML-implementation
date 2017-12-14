import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

# define activation functions and its corressponding derivative
act_funcs = {'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
             'tanh': lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
             'relu': lambda x: np.maximum(x,0)}

act_funcs_der = {'sigmoid': lambda x: act_funcs['sigmoid'](x) * (1 - act_funcs['sigmoid'](x)),
                 'tanh': lambda x: 1 - (act_funcs['tanh'](x))**2,
                 'relu': lambda x: np.array([1 if x[i]>0 else 0 for i in range(x.shape[0])]).reshape(x.shape[0],1)}

# setting Neural Network parameters
S1 = 100 
S2 = 32
S3 = 10
W1 = np.random.normal(0,0.01, [S2,S1])
b1 = np.random.normal(0,0.01, [S2,1])
W2 = np.random.normal(0,0.01, [S3,S2])
b2 = np.random.normal(0,0.01, [S3,1])
params = {'W1': W1, 'b1': b1, 'W2' : W2, 'b2': b2}
# setting activation function
a_func = 'relu'
# setting learning rate
learning_rate = 0.01
# setting regularization parameter lambda
lmda = 0.001
# setting training iterations
iterations = 2500

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

# forward propagation
def FWP(Xi,Yi,F_params,act_func):
    W1, b1, W2, b2 = [F_params[key] for key in F_params.keys()]
    f = act_funcs[act_func]
    z1 = Xi.reshape(Xi.shape[0],1)
    a1 = Xi.reshape(Xi.shape[0],1)
    z2 = W1.dot(a1) + b1
    a2 = f(z2)
    z3 = W2.dot(a2) + b2
    # apply softmax in the output layer
    dom = np.sum(np.exp(z3))
    a3 = np.asarray([np.exp(z3[i,0])/dom for i in range(10)]).reshape(10,1)
    loss = 1 / 2 * np.linalg.norm(a3 - Yi.reshape(Yi.shape[0],1))**2
    re = {'z1': z1, 
          'a1': a1, 
          'z2': z2, 
          'a2': a2, 
          'z3': z3, 
          'a3': a3, 
          'W1': W1, 
          'b1': b1,
          'W2': W2,
          'b2': b2,
          'loss': loss}
    return re

# backpropagation 
def BWP(Yi, B_params, act_func):
    a3, z3, a2, z2, W2, a1 = [B_params[key] for key in ['a3', 'z3', 'a2', 'z2', 'W2', 'a1']]
    d3 = (a3 - Yi.reshape(Yi.shape[0],1))
    dW2 = d3.dot(a2.T)
    db2 = d3
    d2 = W2.T.dot(d3) * act_funcs_der[act_func](z2)
    dW1 = d2.dot(a1.T)
    db1 = d2
    re = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return re

# predict new data point
def predict(Xi, opt_params, act_func):
    W1, b1, W2, b2 = [opt_params[key] for key in opt_params.keys()]
    f = act_funcs[act_func]
    z1 = Xi.reshape(Xi.shape[0],1)
    a1 = Xi.reshape(Xi.shape[0],1)
    z2 = W1.dot(a1) + b1
    a2 = f(z2)
    z3 = W2.dot(a2) + b2
    dom = np.sum(np.exp(z3))
    a3 = np.asarray([np.exp(z3[i,0])/dom for i in range(10)]).reshape(10,1)
    return np.argmax(a3)

# calculating testing error
def testing_error(X, Y, opt_params, act_func):
    error = 0
    N = Y.shape[1]
    for i in range(N):
        Y_pre = predict(X[:,i], opt_params, act_func)
        if np.argmax(Y[:,i]) == Y_pre:
            continue
        error += 1
    return error / N

# Neural Network training function
def train_NN(X, Y, act_func, S1, S2, S3, alpha, lmd, iter):
    N = X.shape[1]
    loss = []
    for i in range(iter):
        loss_new = 0
        # initialize gradients
        dW1 = np.zeros([S2, S1]) 
        db1 = np.zeros([S2, 1])
        dW2 = np.zeros([S3, S2])
        db2 = np.zeros([S3,1])
        
        for j in range(N):
            # apply forward propagation
            FWP_re = FWP(X[:,j], Y[:,j], params, act_func)
            # calculate the current loss value
            loss_new += FWP_re['loss']/N
            # apply backpropagation
            BWP_re = BWP(Y[:,j], FWP_re, act_func)
            dW1 += BWP_re['dW1']
            db1 += BWP_re['db1']
            dW2 += BWP_re['dW2']
            db2 += BWP_re['db2']
        
        # calculate the total loss
        loss_new += lmd/2 * (np.linalg.norm(params['W1'])**2 + np.linalg.norm(params['W2'])**2)            
        loss.append(loss_new)
        
        # calculate gredients    
        dW1 = dW1 / N
        db1 = db1 / N
        dW2 = dW2 / N
        db2 = db2 / N
        
        # update parameters
        params['W1'] = params['W1'] - alpha * (dW1 + lmd * params['W1'])
        params['b1'] = params['b1'] - alpha * db1
        params['W2'] = params['W2'] - alpha * (dW2 + lmd * params['W2'])
        params['b2'] = params['b2'] - alpha * db2
        print('iteration ', i)
    return loss 

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

	# train Neural Network
	loss_v = train_NN(X_train_100, Y_train, a_func, S1, S2, S3, learning_rate, lmda, iterations)

	# output learned parameters
	print(params)

	# print out the activations of the last layer on training data
	for i in range(500):
		output_ = FWP(X_train_100[:,i], Y_train[:,i], params, a_func)
		print(output_['a3'])

	# print out the activations of the last layer on testing data
	for i in range(140):
		output_ = FWP(X_test_100[:,i], Y_test[:,i], params, a_func)
		print(output_['a3'])


	# testing NN on test data set
	print('The classification error on test data is ', testing_error(X_test_100, Y_test, params, a_func))
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

#-------------setting Neural Network parameters---------
S1 = 100
S2 = 64
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
lmda = 0.0001
# setting training iterations
iterations = 1500
#--------------------------------------------------------

#---------------setting auto-encoder parameters----------
a_S1 = 192 * 168 
a_S2 = 100
a_S3 = 192 * 168
a_W1 = np.random.normal(0,0.01, [a_S2,a_S1])
a_b1 = np.random.normal(0,0.01, [a_S2,1])
a_W2 = np.random.normal(0,0.01, [a_S3,a_S2])
a_b2 = np.random.normal(0,0.01, [a_S3,1])
a_params = {'W1': a_W1, 'b1': a_b1, 'W2' : a_W2, 'b2': a_b2}
# setting activation function
a_a_func = 'sigmoid'
# setting learning rate
a_learning_rate = 0.1
# setting regularization parameter lambda
a_lmda = 0.0001
# setting training iterations
a_iterations = 5000
#---------------------------------------------------------

# forward propagation of neural network
def FWP(Xi,Yi,F_params,act_func):
    W1, b1, W2, b2 = [F_params[key] for key in F_params.keys()]
    f = act_funcs[act_func]
    z1 = Xi.reshape(Xi.shape[0],1)
    #print('z1', z1.shape)
    a1 = Xi.reshape(Xi.shape[0],1)
    #print('a1', a1.shape)
    z2 = W1.dot(a1) + b1
    #print('z2',z2.shape)
    a2 = f(z2)
    #print('a2',a2.shape)
    z3 = W2.dot(a2) + b2
    #print('z3',z3.shape)
    # apply softmax in the output layer
    dom = np.sum(np.exp(z3))
    a3 = np.asarray([np.exp(z3[i,0])/dom for i in range(10)]).reshape(10,1)
    #print('a3',a3.shape)
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

# forward propagation of auto-encoder
def FWP_autoencoder(Xi,Yi,F_params,act_func):
    W1, b1, W2, b2 = [F_params[key] for key in F_params.keys()]
    f = act_funcs[act_func]
    z1 = Xi.reshape(Xi.shape[0],1)
    #print('z1', z1.shape)
    a1 = Xi.reshape(Xi.shape[0],1)
    #print('a1', a1.shape)
    z2 = W1.dot(a1) + b1
    #print('z2',z2.shape)
    a2 = f(z2)
    #print('a2',a2.shape)
    z3 = W2.dot(a2) + b2
    #print('z3',z3.shape)
    a3 = f(z3)
    #print('a3',a3.shape)
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

# backpropagation of neural network 
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

# backpropagation of auto-encoder
def BWP_autoencoder(Yi, B_params, act_func):
    a3, z3, a2, z2, W2, a1 = [B_params[key] for key in ['a3', 'z3', 'a2', 'z2', 'W2', 'a1']]
    d3 = (a3 - Yi.reshape(Yi.shape[0],1)) * act_funcs_der[act_func](z3)
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
    m = 64
    for i in range(iter):
        loss_new = 0
        # initialize gradients
        dW1 = np.zeros([S2, S1]) 
        db1 = np.zeros([S2, 1])
        dW2 = np.zeros([S3, S2])
        db2 = np.zeros([S3,1])
        
        sample = random.sample(range(0,500), m)
        batch_X = np.column_stack(X[:, i] for i in sample)
        batch_Y = np.column_stack(Y[:, i] for i in sample)
        
        for j in range(m):
            # apply forward propagation
            FWP_re = FWP(batch_X[:,j], batch_Y[:,j], params, act_func)
            # calculate the current loss value
            loss_new += FWP_re['loss']/m
            # apply backpropagation
            BWP_re = BWP(batch_Y[:,j], FWP_re, act_func)
            dW1 += BWP_re['dW1']
            db1 += BWP_re['db1']
            dW2 += BWP_re['dW2']
            db2 += BWP_re['db2']
        
        # calculate the total loss
        loss_new += lmd/2 * (np.linalg.norm(params['W1'])**2 + np.linalg.norm(params['W2'])**2)            
        loss.append(loss_new)
        
        # calculate gredients    
        dW1 = dW1 / m
        db1 = db1 / m
        dW2 = dW2 / m
        db2 = db2 / m
        
        # update parameters
        params['W1'] = params['W1'] - alpha * (dW1 + lmd * params['W1'])
        params['b1'] = params['b1'] - alpha * db1
        params['W2'] = params['W2'] - alpha * (dW2 + lmd * params['W2'])
        params['b2'] = params['b2'] - alpha * db2
        print('iteration ', i)
    return loss 

# auto-encoder trainning function
def train_encoder(X, Y, act_func, S1, S2, S3, alpha, lmd, iter):
    N = X.shape[1]
    loss_1 = []
    loss_2 = []

    for i in range(iter):
        loss_new = 0
        # initialize gradients
        dW1 = np.zeros([S2, S1]) 
        db1 = np.zeros([S2, 1])
        dW2 = np.zeros([S3, S2])
        db2 = np.zeros([S3,1])
        
        for j in range(N):
            # apply forward propagation
            FWP_re = FWP_autoencoder(X[:,j], Y[:,j], a_params, act_func)
            # calculate the current loss value
            loss_new += FWP_re['loss']/N
            # apply backpropagation
            BWP_re = BWP_autoencoder(Y[:,j], FWP_re, act_func)
            dW1 += BWP_re['dW1']
            db1 += BWP_re['db1']
            dW2 += BWP_re['dW2']
            db2 += BWP_re['db2']
        
        # calculate the total loss
        loss_1.append(loss_new)
        loss_new += lmd/2 * (np.linalg.norm(a_params['W1'])**2 + np.linalg.norm(a_params['W2'])**2)            
        loss_2.append(loss_new)
        
        # calculate gredients    
        dW1 = dW1 / N
        db1 = db1 / N
        dW2 = dW2 / N
        db2 = db2 / N
        
        # update parameters
        a_params['W1'] = a_params['W1'] - alpha * (dW1 + lmd * a_params['W1'])
        a_params['b1'] = a_params['b1'] - alpha * db1
        a_params['W2'] = a_params['W2'] - alpha * (dW2 + lmd * a_params['W2'])
        a_params['b2'] = a_params['b2'] - alpha * db2
        print('iteration ', i)
    return loss_1, loss_2

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

	# train autoencoder to reduce the dimension of the dataset
	loss_v1, loss_v2 = train_encoder(X_train, X_train, a_a_func, a_S1, a_S2, a_S3, a_learning_rate, a_lmda, a_iterations)

	# output learned parameters of auto-encoder
	print(a_params)

	# print out the activations of the last layer on training data of auto-encoder
	for i in range(500):
		output_ = FWP_autoencoder(X_train[:,i], X_train[:,i], a_params, a_a_func)
		print(output_['a3'])

	# print out the activations of the last layer on testing data of auto-encoder
	for i in range(140):
		output_ = FWP_autoencoder(X_test[:,i], X_test[:,i], a_params, a_a_func)
		print(output_['a3'])


	# generate the low-dimensional data using the parameters learned by auto-encoder
	X_train_100 = np.column_stack(FWP_autoencoder(X_train[:,i], X_train[:,i], a_params, a_a_func)['a2'] for i in range(500))
	X_test_100 = np.column_stack(FWP_autoencoder(X_test[:,i], X_test[:,i], a_params, a_a_func)['a2'] for i in range(140))

    # train Neural Network using the low-dimensional data
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

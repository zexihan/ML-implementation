import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

act_funcs = {'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
             'tanh': lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
             'relu': lambda x: np.maximum(x,0)}

act_funcs_der = {'sigmoid': lambda x: act_funcs['sigmoid'](x) * (1 - act_funcs['sigmoid'](x)),
                 'tanh': lambda x: 1 - (act_funcs['tanh'](x))**2,
                 'relu': lambda x: np.array([1 if x[i]>0 else 0 for i in range(x.shape[0])]).reshape(x.shape[0],1)}

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
	I = np.identity(10)
	Y_train = np.column_stack(I[:,i] for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	Y_test = np.column_stack(I[:,i] for i in range(10) for j in range(14))
	X_train = X_train / 255
	X_test = X_test / 255

	# train autoencoder to reduce the dimension of the dataset
	loss_v1, loss_v2 = train_encoder(X_train, X_train, a_a_func, a_S1, a_S2, a_S3, a_learning_rate, a_lmda, a_iterations)

	# generate the low-dimensional data using the parameters learned by auto-encoder
	X_train_100 = np.column_stack(FWP_autoencoder(X_train[:,i], X_train[:,i], a_params, a_a_func)['a2'] for i in range(500))
	X_test_100 = np.column_stack(FWP_autoencoder(X_test[:,i], X_test[:,i], a_params, a_a_func)['a2'] for i in range(140))
	X_train = X_train_100.T
	X_test = X_test_100.T

	# train 1 vs all decision boundaries 
	decision_boundaries = []
	alpha = 0.0001
	iters = 1000
	for i in range(10):
		Y_train = np.zeros([X_train.shape[0],1])
		Y_test = np.zeros([X_test.shape[0], 1])
		omega = np.zeros((X_train.shape[1],1))
		Y_train[i*50:(i+1)*50,0] = 1
		Y_test[i*14:(i+1)*14,0] = 1
		omega = gradientDescent(X_train/255, Y_train, omega, alpha, iters, 0.001)
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


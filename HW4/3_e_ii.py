import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc

# define activation functions and its corressponding derivative
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

    #----------SVM SMO-----------------------------
def classification_error(X,omega, bias,Y):
    misclassify = 0
    re = np.dot(X,omega) + bias
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != -1):
            misclassify += 1
    return misclassify / len(re)

def hypothesis_func(idx, alphas, X, Y, b):
    kernel = np.dot(X,X[idx].reshape(100,1))
    return np.dot(alphas.T,Y*kernel) + b

def randj(idx, length):
    j = idx
    while j == idx:
        j = np.random.randint(length)
    return j

def clip_alphaj(L, H, alphaj):
    if alphaj > H:
        return H
    elif alphaj < L:
        return L
    else:
        return alphaj

def simplified_SMO(C, tol, maxpasses, X, Y):
    alpha = np.zeros((X.shape[0], 1))
    bias = 0
    passes = 0
    while (passes < maxpasses):
        num_changed_alphas = 0
        for i in range(len(alpha)):
            # calculate E_ip
            E_i = hypothesis_func(i, alpha, X, Y, bias) - Y[i]
            if (Y[i] * E_i < -tol and alpha[i] < C) or (Y[i] * E_i > tol and alpha[i] > 0):
                # randomly select j not equal i
                j = randj(i, len(alpha))
                # calculate E_j
                E_j = hypothesis_func(j, alpha, X, Y, bias) - Y[j]
                # save old alphas
                alphai_old = alpha[i].copy()
                alphaj_old = alpha[j].copy()
                # compute L and H
                if Y[i] != Y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C+alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                # if L==H continue
                if L == H:
                    continue
                # compute eta
                eta = 2.0 * np.dot(X[i],X[j].T) - np.dot(X[i],X[i].T) - np.dot(X[j],X[j].T)
                if eta >= 0:
                    print('eta is bigger than 0')
                    continue
                # compute and clip new value for alpha_j
                #print('old alphaj is ', alphaj_old)
                alpha[j] = alpha[j] - (Y[j] * (E_i - E_j)) / eta
                #print('new alphaj is ', alpha[j])
                #print('L is ', L)
                #print('H is ', H)
                alpha[j] = clip_alphaj(L,H,alpha[j])
                #print('clipped alphaj is ', alpha[j])
                #print('check alphaj_old is ', alphaj_old)
                
                if abs(alpha[j] - alphaj_old) < 10**(-5):
                    #print('alpha is not moving enough')
                    continue
                # determine value for alpha_i
                alpha[i] = alpha[i] + Y[i]*Y[j]*(alphaj_old - alpha[j])
                # compute b_1 and B_2
                b_1 = bias - E_i - Y[i]*(alpha[i] - alphai_old)*np.dot(X[i],X[i].T) - Y[j]*(alpha[j] - alphaj_old)*np.dot(X[i],X[j].T)
                b_2 = bias - E_j - Y[i]*(alpha[i] - alphai_old)*np.dot(X[i],X[j].T) - Y[j]*(alpha[j] - alphaj_old)*np.dot(X[j],X[j].T)
                # compute b
                if 0 < alpha[i] and alpha[i] < C:
                    bias = b_1
                elif 0 < alpha[j] and alpha[j] < C:
                    bias = b_2
                else:
                    bias = (b_1 + b_2) / 2.0
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
        #print('iteration', passes)
    return alpha, bias

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
	dec_b = []
	for i in range(10):
		Y_train = np.ones([X_train.shape[0],1]) * -1
		Y_test = np.ones([X_test.shape[0], 1]) * -1
		Y_train[i*50:(i+1)*50,0] = 1
		Y_test[i*14:(i+1)*14,0] = 1
		a, b = simplified_SMO(0.5, 0.001, 200, X_train, Y_train)
		omega = (sum(a * Y_train * X_train)).reshape(X_train.shape[1],1)
		print(classification_error(X_train, omega, b, Y_train))
		print(classification_error(X_test, omega, b, Y_test))
		dec_b.append([omega,b])
		print('Finish training for person ', i)
		
	# test classifier on testing data
	omega_matrix = np.column_stack(dec_b[i][0] for i in range(10))
	omega_norms = np.column_stack(np.linalg.norm(omega_matrix[:,i]) for i in range(10))
	b_vector = np.column_stack(dec_b[i][1] for i in range(10))
	res = (np.dot(X_test, omega_matrix) + b_vector) / omega_norms

	error = 0
	label = 0
	for i in range(140):
		if i!=0 and i%14 == 0:
			label += 1
		if np.argmax(res[i,:]) != label:
			error += 1
	print('The classification error on test data is ', error / 140)
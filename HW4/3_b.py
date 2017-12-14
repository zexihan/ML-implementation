import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc

#----------SVM SMO-----------------------------------------------
def classification_error(X,omega, bias,Y):
    misclassify = 0
    re = np.dot(X,omega) + bias
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != -1):
            misclassify += 1
    return misclassify / len(re)

def hypothesis_func(idx, alphas, X, Y, b):
    kernel = np.dot(X,X[idx].reshape(192*168,1))
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

	# reshape and normalize data
	X_train = np.column_stack(train_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	X_train = X_train.T / 255
	X_test = X_test.T / 255

	# collect decision boundaries
	dec_b = []
	# train 1 vs all classifier
	for i in range(10):
		Y_train = np.ones([X_train.shape[0],1]) * -1
		Y_test = np.ones([X_test.shape[0], 1]) * -1
		Y_train[i*50:(i+1)*50,0] = 1
		Y_test[i*14:(i+1)*14,0] = 1
		a, b = simplified_SMO(50, 0.001, 200, X_train, Y_train)
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
	print('Classification error on test data set is ', error / 140)

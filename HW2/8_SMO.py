import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

def classification_error(X,omega, bias,Y):
    misclassify = 0
    re = np.dot(X,omega) + bias
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != -1):
            misclassify += 1
    return misclassify / len(re)

def hypothesis_func(idx, alphas, X, Y, b):
    kernel = np.dot(X,X[idx].reshape(2,1))
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
                alpha[j] = alpha[j] - (Y[j] * (E_i - E_j)) / eta
                alpha[j] = clip_alphaj(L,H,alpha[j])
                
                if abs(alpha[j] - alphaj_old) < 10**(-5):
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
    return alpha, bias

if __name__ == '__main__':
	datasets = ['data1.mat', 'data2.mat']
	for j in range(2):
		data = sio.loadmat(datasets[j])
		X_train = data['X_trn']
		Y_trn = data['Y_trn']
		X_test = data['X_tst']
		Y_tst = data['Y_tst']
		Y_train = np.ones((Y_trn.shape))
		Y_test = np.ones((Y_tst.shape))
		for i in range(len(Y_train)):
		    if Y_trn[i] == 0:
		        Y_train[i] = -1
		    if Y_trn[i] == 1:
		        break
		for i in range(len(Y_test)):
		    if Y_tst[i] == 0:
		        Y_test[i] = -1
		    if Y_tst[i] == 1:
		        break 
		C = np.linspace(0.1,50,100)
		error_train = []
		error_test = []
		for c in C:
			a, b = simplified_SMO(c, 0.001, 500, X_train, Y_train)
			omega = (sum(a * Y_train * X_train)).reshape(2,1)
			error_train.append(classification_error(X_train, omega, b, Y_train))
			error_test.append(classification_error(X_test, omega, b, Y_test))

		fig, ax = plt.subplots(1,2,figsize=(16,6))
		ax[0].plot(C, error_train)
		ax[0].set_xlabel('regularization parameter C')
		ax[0].set_ylabel('classification error')
		if j == 0:
			ax[0].set_title('Classification error on dataset1 training set')
		else:
			ax[0].set_title('Classification error on dataset2 training set')
		ax[1].plot(C, error_test)
		ax[1].set_xlabel('regularization parameter C')
		ax[1].set_ylabel('classification error')
		if j == 0:
			ax[1].set_title('Classification error on dataset1 testing set')
		else:
			ax[1].set_title('Classification error on dataset2 testing set')

		x = np.linspace(min(X_train[:,0]), max(X_train[:,0]), 100)
		f = (-b - omega[0] * x) / omega[1]
		f_left = (-1 - b - omega[0] * x) / omega[1]
		f_right = (1 - b - omega[0] * x) / omega[1]
		f = f[0,:]
		f_left = f_left[0,:]
		f_right = f_right[0,:]
		fig2, ax = plt.subplots(figsize=(10,8))
		for i in range(len(Y_train)):
			if Y_train[i] != -1:
				break
		ax.scatter(X_train[0:i-1,0], X_train[0:i-1,1], c='r', marker = '+', label='training data class 0')
		ax.scatter(X_train[i:,0], X_train[i:,1], c='b', marker = '+', label='training data class 1')
		ax.plot(x, f, 'g', label = 'decision boundary')
		ax.plot(x, f_left, 'y--', label = 'margin')
		ax.plot(x, f_right, 'y--')
		for i in range(len(Y_train)):
			if Y_test[i] != -1:
				break
		ax.scatter(X_test[0:i-1,0], X_test[0:i-1,1], c='r', marker = '^', label='testing data class 0')
		ax.scatter(X_test[i:,0], X_test[i:,1], c='b', marker = '^', label='testing data class 1')
		ax.legend(loc=2)
	plt.show()

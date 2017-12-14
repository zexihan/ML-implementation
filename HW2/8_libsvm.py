import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

def classification_error(X,omega, bias,Y):
    misclassify = 0
    re = np.dot(X,omega) + bias
    for i in range(len(re)):
        if (re[i] > 0 and Y[i] != 1) or (re[i] < 0 and Y[i] != -1):
            misclassify += 1
    return misclassify / len(re)

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

		C = np.linspace(0.1,100,200)
		error_train = []
		error_test = []

		for c in C:
		    clf = svm.SVC(kernel='linear', C=c)
		    clf.fit(X_train, Y_train[:,0])
		    omega = clf.coef_[0]
		    b = clf.intercept_[0]
		    error_test.append(classification_error(X_test, omega, b, Y_test))
		    error_train.append(classification_error(X_train, omega, b, Y_train))

		supportvector = clf.support_
		left = [i for i in supportvector if Y_train[i] == -1]
		right = [i for i in supportvector if Y_train[i] == 1]

		fig1, ax = plt.subplots(1,2,figsize=(16,6))
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

		fig2, ax = plt.subplots(figsize=(10,8))
		x = np.linspace(min(X_train[:,0]), max(X_train[:,0]), 100)
		f = (-b - omega[0] * x) / omega[1]
		f_left = (np.dot(X_train[left[0]], omega) + b - b - omega[0] * x) / omega[1]
		f_right = (np.dot(X_train[right[0]], omega) + b - b - omega[0] * x) / omega[1]

		for i in range(len(Y_train)):
		    if Y_train[i] != -1:
		        break
		ax.scatter(X_train[0:i-1,0], X_train[0:i-1,1], c='r', marker = '+', label='training data class 0')
		ax.scatter(X_train[i:,0], X_train[i:,1], c='b', marker = '+', label='training data class 1')
		ax.plot(x, f, 'g', label = 'decision boundary')
		ax.plot(x, f_left, 'y--', label = 'margin')
		ax.plot(x, f_right, 'y--')
		for i in range(len(Y_test)):
			if Y_test[i] != -1:
				break
		ax.scatter(X_test[0:i-1,0], X_test[0:i-1,1], c='r', marker = '^', label='testing data class 0')
		ax.scatter(X_test[i:,0], X_test[i:,1], c='b', marker = '^', label='testing data class 1')
		ax.legend(loc=2)
	plt.show()
	
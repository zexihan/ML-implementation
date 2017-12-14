import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#-------------PCA-----------------
def reg_PCA(Y_train, d):
    row, col = Y_train.shape
    mu_bar = Y_train.mean(1).reshape(row,1)
    Y_train_temp = Y_train - mu_bar
    U,S,V = np.linalg.svd(Y_train_temp)
    return np.dot(U[:,0:d].T, Y_train_temp)

if __name__ == '__main__':
	# load data
	data = sio.loadmat('ExtYaleB10.mat')
	train_samples = data['train']
	test_samples = data['test']

	# reshape the data
	X_train = np.column_stack(train_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	X_train = X_train / 255
	X_test = X_test / 255
	dataset = X_train
	for i in range(10):
		dataset = np.insert(dataset, [i*50+i*14], X_test[:,i*14:(i+1)*14], axis=1)

	# apply PCA
	dataset_2 = reg_PCA(dataset, 2)

	# plot results
	fig1, ax1 = plt.subplots(figsize=(10,8))
	colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'b']
	markers = ['o', '^', 's', '*', '+', 'd', '|', '>', 'x', 'h']
	for i in range(10):
		ax1.scatter(dataset_2[0,i*64:(i+1)*64],dataset_2[1,i*64:(i+1)*64], marker=markers[i], facecolors=colors[i], edgecolors='k')
	ax1.set_title('2-dimensional data after PCA ')

	plt.show()

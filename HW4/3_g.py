import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

#------------PCA------------------------
def reg_PCA(Y, d):
    row, col = Y.shape
    mu_bar = Y.mean(1).reshape(row,1)
    Y_temp = Y - mu_bar
    U,S,V = np.linalg.svd(Y_temp)
    return np.dot(U[:,0:d].T, Y_temp)
#---------------------------------------

#-----------------kmeans----------------
def kmeans(Y,k):
    # Initialize centroids randomly
    centroids = getRandomCentroids(Y,k)
    old_centroids = None
    while not convergence(old_centroids, centroids):
        # save old centroids for convergence test
        old_centroids = centroids
        # assign data points to new centroids
        labels = getlabels(Y, centroids, k)
        # generate new centroids
        centroids = getcentroids(Y, labels, k)
    return centroids, labels

def getRandomCentroids(Y,k):
    idex = np.column_stack(np.random.randint(i, (i+1)*64, size=1) for i in range(k))
    center = np.zeros([Y.shape[0], k])
    for i in range(k):
        center[:,i] = Y[:, idex[0,i]]
    return center

def convergence(old_centers, new_centers):
    return np.array_equal(old_centers, new_centers)

def getlabels(Y, centers, k):
    row,col = Y.shape
    labels = np.ones([1, col], dtype=np.int)
    for i in range(col):
        old_d = float("inf")
        for j in range(k):
            new_d = np.linalg.norm(Y[:,i] - centers[:,j])**2
            if new_d < old_d:
                labels[0,i] = j
                old_d = new_d
    return labels

def getcentroids(Y, labels, k):
    row,col = Y.shape
    new_center = np.zeros([row, k])
    for i in range(k):
        datapoints = [j for j in range(col) if labels[0,j] == i]
        for m in datapoints:
            new_center[:,i] = new_center[:,i] + Y[:,m]
        new_center[:,i] = new_center[:,i] / len(datapoints)
    return new_center

if __name__ == '__main__':
	# load data
	data = sio.loadmat('ExtYaleB10.mat')
	train_samples = data['train']
	test_samples = data['test']

	# reshape data
	X_train = np.column_stack(train_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	X_train = X_train 
	X_test = X_test 

	# combining train and test dataset
	dataset = X_train
	for i in range(10):
		dataset = np.insert(dataset, [i*50+i*14], X_test[:,i*14:(i+1)*14], axis=1)
	I = np.identity(10)
	human_label = np.column_stack(I[:,i] for i in range(10) for j in range(64))

	# apply PCA to reduce data to 100 dimension
	dataset_100 = reg_PCA(dataset, 100)

	# apply PCA to reduce data to 2 dimension
	dataset_2 = reg_PCA(dataset, 2)

	# apply Kmeans to original data and plot the result in 2-D data
	centroids, labels = kmeans(dataset, 10)
	fig1, ax1 = plt.subplots(figsize=(10,8))
	colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
	for i in range(640):
		ax1.scatter(dataset_2[0,i],dataset_2[1,i], facecolors= colors[labels[0,i]], edgecolors='k')
	ax1.set_title('The result of applying Kmeans to original data')

	# calculating clustering error
	cluster_label = np.zeros([10, 640])
	for i in range(640):
		cluster_label[labels[0,i],i] = 1
	correct = 0
	for i in range(640):
		for j in range(640):
			if j!=i and (np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 2 or np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 0):
				correct += 1
	print('The clustering error on the original dataset is ', 1 - correct / (640*639))

	# apply Kmeans to 100 dimensional data and plot the result in 2-D data
	centroids, labels = kmeans(dataset_100, 10)
	fig2, ax2 = plt.subplots(figsize=(10,8))
	colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
	for i in range(640):
		ax2.scatter(dataset_2[0,i],dataset_2[1,i], facecolors= colors[labels[0,i]], edgecolors='k')
	ax2.set_title('The result of applying Kmeans to 100 dimensional data')

	# calculating clustering error
	cluster_label = np.zeros([10, 640])
	for i in range(640):
		cluster_label[labels[0,i],i] = 1
	correct = 0
	for i in range(640):
		for j in range(640):
			if j!=i and (np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 2 or np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 0):
				correct += 1
	print('The clustering error on the 100-D dataset is ', 1 - correct / (640*639))

	plt.show()
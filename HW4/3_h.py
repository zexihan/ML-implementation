import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

#---------------kmeans-------------
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
    #idex = np.random.randint(Y.shape[1], size=k)
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
#-------------------------------------------------------

#------------------Spectral Clustering-----------------
def nn_matrix(Y):
    row, col = Y.shape
    matrix = np.zeros([col, col])
    for i in range(col):
        for j in range(col):
            matrix[i,j] = np.linalg.norm(Y[:,i] - Y[:,j])
    return matrix

def Spectral_clustering(Y, sigma, knn, k):
    simi_M = nn_matrix(Y)
    W = np.zeros(simi_M.shape)
    # creat similarity matrix W based on knn
    for i in range(simi_M.shape[0]):
        sorted_id = simi_M[i,:].argsort()
        for j in range(knn):
            W[i,sorted_id[j]] = np.exp(-1/(2*sigma**2)*simi_M[i,sorted_id[j]]**2)
    # build laplacian matrix L
    D = np.zeros(W.shape)
    for i in range(D.shape[0]):
        D[i,i] =  np.sum(W[i,:])
    L = D - W
    # calculate k bottom eigenvectors of L
    eigvals, eigvects = np.linalg.eig(L)
    sorted_id = eigvals.argsort()
    V = np.column_stack(eigvects[:,sorted_id[i]] for i in range(k))
    # apply Kmeans to rows of V
    centroids, labels = kmeans(V.T, k)
    return V.T, labels    
#----------------------------------------------------------

if __name__ == '__main__':
	# load data
	data = sio.loadmat('ExtYaleB10.mat')
	train_samples = data['train']
	test_samples = data['test']

	# reshape and combine training and testing data
	X_train = np.column_stack(train_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(50))
	X_test = np.column_stack(test_samples[0,i][:,:,j].reshape(192*168,1) for i in range(10) for j in range(14))
	X_train = X_train/255
	X_test = X_test /255
	dataset = X_train
	for i in range(10):
		dataset = np.insert(dataset, [i*50+i*14], X_test[:,i*14:(i+1)*14], axis=1)
	I = np.identity(10)
	human_label = np.column_stack(I[:,i] for i in range(10) for j in range(64))

	# apply spectral clustering on dataset
	vectors, labels = Spectral_clustering(dataset, 1, 20, 10)

	# calculate and print out the clustering error
	cluster_label = np.zeros([10, 640])
	for i in range(640):
		cluster_label[labels[0,i],i] = 1
	correct = 0
	for i in range(640):
		for j in range(640):
			if j!=i and (np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 2 or np.dot(human_label[:,i].T, human_label[:,j]) + np.dot(cluster_label[:,i].T, cluster_label[:,j]) == 0):
				correct += 1
	print('The clustering error is ', 1 - correct / (640*639))
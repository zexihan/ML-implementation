import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#-----------Regular PCA-------------
def reg_PCA(Y, d):
	# standardization
    row, col = Y.shape
    mu_bar = Y.mean(1).reshape(row,1)
    Y_temp = Y - mu_bar
    # svd decomposition
    U,S,V = np.linalg.svd(Y_temp)
    return np.dot(U[:,0:d].T, Y_temp)
#------------------------------------

#-----------Kmeans-------------------
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
    idex = np.random.randint(Y.shape[1], size = k)
    center = np.zeros([Y.shape[0], k])
    for i in range(k):
        center[:,i] = Y[:, idex[i]]
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
#--------------------------------------

if __name__ == '__main__':
	# load data
	data = sio.loadmat('dataset1.mat')
	Y_samples = data['Y']

	# 7. Part A i
	fig1, ax1 = plt.subplots(figsize=(10,8))
	ax1.scatter(Y_samples[0,:],Y_samples[1,:], facecolors='r', edgecolors='k')
	ax1.set_xlabel('First feature of Y')
	ax1.set_ylabel('Second feature of Y')
	ax1.set_title('7 Part A i: data points scatter')
	plt.show()

	# 7. Part A ii
	fig2, ax2 = plt.subplots(figsize=(10,8))
	ax2.scatter(Y_samples[1,:],Y_samples[2,:], facecolors='r', edgecolors='k')
	ax2.set_xlabel('Second feature of Y')
	ax2.set_ylabel('Third feature of Y')
	ax2.set_title('7 Part A ii: data points scatter')
	plt.show()

	# 7. Part A iii
	X_samples = reg_PCA(Y_samples, 2)
	fig3, ax3 = plt.subplots(figsize=(10,8))
	ax3.scatter(X_samples[0,:],X_samples[1,:], facecolors='r', edgecolors='k')
	ax3.set_xlabel('First feature of the 2-D data after PCA')
	ax3.set_ylabel('Second feature of the 2-D data after PCA')
	ax3.set_title('7 Part A iii: data points scatter after PCA')
	plt.show()

	# 7. Part A iv
	centroids, labels = kmeans(Y_samples, 2)
	fig4, ax4 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax4.scatter(X_samples[0,i],X_samples[1,i], facecolors='r', edgecolors='k')
		else:
			ax4.scatter(X_samples[0,i],X_samples[1,i], facecolors='b', edgecolors='k')
	ax4.set_xlabel('First feature of the 2-D data after PCA')
	ax4.set_ylabel('Second feature of the 2-D data after PCA')
	ax4.set_title('7 Part A iv: Result of applying Kmeans to 40 dimensional data')
	plt.show()

	# 7. Part A v
	centroids, labels = kmeans(X_samples, 2)
	fig5, ax5 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax5.scatter(X_samples[0,i],X_samples[1,i], facecolors='r', edgecolors='k')
		else:
			ax5.scatter(X_samples[0,i],X_samples[1,i], facecolors='b', edgecolors='k')
	ax5.set_xlabel('First feature of the 2-D data after PCA')
	ax5.set_ylabel('Second feature of the 2-D data after PCA')
	ax5.set_title('7 Part A v: Result of applying Kmeans to 2 dimensional data')
	plt.show()


	
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#-------------------Regular PCA-----------------------
def reg_PCA(Y, d):
	# standardization
    row, col = Y.shape
    mu_bar = Y.mean(1).reshape(row,1)
    Y_temp = Y - mu_bar
    # svd decomposition
    U,S,V = np.linalg.svd(Y_temp)
    return np.dot(U[:,0:d].T, Y_temp)
#-----------------------------------------------------

#--------------------Kmeans---------------------------
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
#-----------------------------------------------------

#-----------Kernel PCA--------------------------------
def K_PCA(Y, d, sigma):
    # step 1 calculating kernel matrix
    K = kernel_matrix(Y,sigma)
    # step 2
    one_N = np.ones(K.shape)/K.shape[0]
    K_bar = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)
    # step 3 Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.
    eigvals, eigvects = np.linalg.eig(K_bar)
    sorted_id = eigvals.argsort()
    sorted_id = sorted_id[::-1]
    # step 4 Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    W = np.column_stack(eigvects[:,sorted_id[i]]/np.sqrt(eigvals[sorted_id[i]]) for i in range(d))
    return np.dot(W.T, K_bar)

def kernel_matrix(Y, sigma):
    row, col = Y.shape
    gamma = -1 / (2 * sigma**2)
    matrix = np.zeros([col, col])
    for i in range(col):
        for j in range(col):
            matrix[i,j] = np.exp(gamma * (np.linalg.norm(Y[:,i] - Y[:,j])**2))
    return matrix
#-----------------------------------------------------

#-----------Spectral Clustering-----------------------
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

def nn_matrix(Y):
    row, col = Y.shape
    matrix = np.zeros([col, col])
    for i in range(col):
        for j in range(col):
            matrix[i,j] = np.linalg.norm(Y[:,i] - Y[:,j])
    return matrix
#------------------------------------------------------

if __name__ == '__main__':
	# load data
	data = sio.loadmat('dataset2.mat')
	Y_samples = data['Y']

	# 7. Part B i
	fig1, ax1 = plt.subplots(figsize=(10,8))
	ax1.scatter(Y_samples[0,:],Y_samples[1,:], facecolors='r', edgecolors='k')
	ax1.set_xlabel('First feature of Y')
	ax1.set_ylabel('Second feature of Y')
	ax1.set_title('7 Part B i: data points scatter')
	plt.show()

	# 7. Part B ii
	fig2, ax2 = plt.subplots(figsize=(10,8))
	ax2.scatter(Y_samples[1,:],Y_samples[2,:], facecolors='r', edgecolors='k')
	ax2.set_xlabel('Second feature of Y')
	ax2.set_ylabel('Third feature of Y')
	ax2.set_title('7 Part B ii: data points scatter')
	plt.show()

	# 7. Part B iii
	X_samples = reg_PCA(Y_samples, 2)
	fig3, ax3 = plt.subplots(figsize=(10,8))
	ax3.scatter(X_samples[0,:],X_samples[1,:], facecolors='r', edgecolors='k')
	ax3.set_xlabel('First feature of the 2-D data after PCA')
	ax3.set_ylabel('Second feature of the 2-D data after PCA')
	ax3.set_title('7 Part B iii: data points scatter after PCA')
	plt.show()

	# 7. Part B iv
	centroids, labels = kmeans(X_samples, 2)
	fig4, ax4 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax4.scatter(X_samples[0,i],X_samples[1,i], facecolors='r', edgecolors='k')
		else:
			ax4.scatter(X_samples[0,i],X_samples[1,i], facecolors='b', edgecolors='k')
	ax4.set_xlabel('First feature of the 2-D data after PCA')
	ax4.set_ylabel('Second feature of the 2-D data after PCA')
	ax4.set_title('7 Part B iv: Result of applying Kmeans to 2 dimensional data obtained by PCA')
	plt.show()

	# 7. Part B v
	X_samples = K_PCA(Y_samples,2,10)
	centroids, labels = kmeans(X_samples,2)
	fig5, ax5 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax5.scatter(X_samples[0,i],X_samples[1,i], facecolors='r', edgecolors='k')
		else:
			ax5.scatter(X_samples[0,i],X_samples[1,i], facecolors='b', edgecolors='k')
	ax5.set_xlabel('First feature of the 2-D data after Kernel-PCA')
	ax5.set_ylabel('Second feature of the 2-D data after Kernel-PCA')
	ax5.set_title('7 Part B v: Result of applying Kmeans to 2 dimensional data obtained by Kernel-PCA')
	plt.show()

	# 7. Part B vi
	# Apply spectral clustering to reducing the original 40-D data to 2-D data and then apply 
	# Kmeans to separate the data to two groups
	vectors, labels = Spectral_clustering(Y_samples, 1, 10, 2)
	fig6, ax6 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax6.scatter(vectors[0,i],vectors[1,i], facecolors='r', edgecolors='k')
		else:
			ax6.scatter(vectors[0,i],vectors[1,i], facecolors='b', edgecolors='k')
	ax6.set_xlabel('First feature of the bottom vectors obtained by Spectral')
	ax6.set_ylabel('Second feature of the bottom vectors obtained by Spectral')
	ax6.set_title('7 Part B vi: Result of applying Spectral Clustering to the original data')
	plt.show()

	# Prove the result of Spectral Clustering by drawing the separated groups of data on the 2-D data obtained by PCA
	fig7, ax7 = plt.subplots(figsize=(10,8))
	for i in range(labels.shape[1]):
		if labels[0,i] == 0:
			ax7.scatter(X_samples[0,i],X_samples[1,i], facecolors='r', edgecolors='k')
		else:
			ax7.scatter(X_samples[0,i],X_samples[1,i], facecolors='b', edgecolors='k')
	ax7.set_xlabel('First feature of the 2-D data after PCA')
	ax7.set_ylabel('Second feature of the 2-D data after PCA')
	ax7.set_title('Prove the correctness of the result of Spectral Clustering')
	plt.show()

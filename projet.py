# -*- coding: utf-8 -*-
#% Configuration
rfSize = 6
numCentroids = 400
numPatches = 400000
CIFAR_DIM = [32,32,3]
c = 100
nb_kmeans = 20


from six.moves import cPickle
import numpy as np
import os
from pprint import pprint
import sys
from six.moves import cPickle
from six.moves import range
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing
import scipy as sp
from scipy.sparse import coo_matrix
from matplotlib import pyplot
import math
from joblib import Parallel, delayed
import multiprocessing
import itertools
from sklearn.feature_extraction import image

import time

import progressbar

def load_batch(fpath, label_key='labels'):
	
	f = open(fpath, 'rb')
	if sys.version_info < (3,): #si python 2 pas de soucis, sinon py3 decode
		d = cPickle.load(f)
	else:
		d = cPickle.load(f, encoding="bytes")
		# decode utf8
		for k, v in d.items():
			del(d[k])
			d[k.decode("utf8")] = v
	f.close()
	data = d["data"]

	labels = d[label_key]

	data = data.reshape(data.shape[0], 3072)
	return data, labels
	
def load_data():
	"""
	Permet de charger les donnees et les transformer en donnees de train et test
	"""

	dirname = "cifar-10-batches-py"

	nb_train_samples = 50000

	X_train = np.zeros((nb_train_samples, 3072), dtype="uint8")
	y_train = np.zeros((nb_train_samples,), dtype="uint8")

	for i in range(1, 6):
		fpath = os.path.join(dirname, 'data_batch_' + str(i))
		data, labels = load_batch(fpath)
		
		X_train[(i-1)*10000:i*10000, :] = data #50 000 premieres = train
		y_train[(i-1)*10000:i*10000] = labels #50 000 premieres = train

	fpath = os.path.join(dirname, 'test_batch') 
	X_test, y_test = load_batch(fpath)

	y_train = np.reshape(y_train, (len(y_train), 1))

	#on va chercher les donnees dans le test_batch
	y_test = np.reshape(y_test, (len(y_test), 1)) 

	return (X_train, y_train), (X_test, y_test)
	
def kmeans(X=None, k=None, iterations=None):
	
	centroids = np.random.randn(k, np.size(X, axis=1)) * 0.1 #Initialise les centroids au hasard
	
	BATCH_SIZE = 1000 #nombre d'éléments par iteration
	
	progress = progressbar.ProgressBar(widgets=['K-means iterations',progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
	for itr in progress(range(1,iterations+1)):
		time.sleep(0.001)
		progress.update(itr)

		c2 = 0.5 * np.sum(centroids ** 2,axis=1, keepdims=True) # c2.shape = (K,1) somme des carres à minimiser
		
		summation = np.zeros((k, np.size(X, axis=1))) #summation.shape = (K, ) Mean est non associatif, donc décompose le calcul en 2 parties associatives : la somme et le compteur
		counts = np.zeros((k, 1))
		
		i = 0
		while i < np.size(X, 0):	
			lastIndex = min(i + BATCH_SIZE, np.size(X, 0))
			m = lastIndex - i
			
			a = np.dot(centroids,np.transpose(X[i:lastIndex, :])) #a.shape = (K, BATCH_SIZE)
			b = a - c2 #b.shape = (K, BATCH_SIZE) - (K,1)

			val = np.max(b,axis=0,keepdims=True) # On recupere la valeur maximale de chaque colonne
			labels = np.argmax(b,axis=0) # On recupere la position de la valeur maximale

			S = np.zeros((BATCH_SIZE, k)) #on crée une matrice creuse (sparse matrix) avec bcp de zeros 1000 * K
			S[range(BATCH_SIZE), labels] = 1 #On met un 1 à chaque position trouvée (càd 1 si x ∈ argmin mu(z) − z_i, 0 sinon ), chaque donnée est assignée à la classe du centre le plus proche
			
			summation = summation + np.dot(np.transpose(S),X[i:lastIndex, :]) 
			tp = np.transpose(np.sum(S,0, keepdims=True))
			counts = counts + tp
			i += BATCH_SIZE
			
		with np.errstate(invalid='ignore'):
			centroids = summation / counts #on met à jour les moyennes
		
		where_are_NaNs = np.isnan(centroids)
		centroids[where_are_NaNs] = 0
		
		assert not np.any(np.isnan(centroids))
		
	return centroids

	
    
def normalize(x):
    """
    On normalise chaque patch en soustrayant la moyenne et en divisant par la variance pour réduire le constraste
    """
    temp1 = x - x.mean(1, keepdims=True)
    temp2 = np.sqrt(x.var(1, keepdims=True) + 10)

    return temp1 / temp2

def getPatch(numPatches, Xtrain, patch_size):
	
	patches = np.zeros((numPatches, rfSize * rfSize * 3), dtype="uint8")  #shape 50000, 32 * 32 * 3
	Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],32,32,3)) #On reshape les données d'entrées pour extraire les patches
		
	progress = progressbar.ProgressBar(widgets=['Extracting patches',progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
	for i in progress(range(numPatches)):
		progress.update(i)
		
		row = np.random.randint(CIFAR_DIM[0] - rfSize + 1) # distribution uniforme 
		col = np.random.randint(CIFAR_DIM[1] - rfSize + 1) 
		
		img = Xtrain[i %  Xtrain.shape[0]]+1 #on extrait numPatches/50000 = n patch par images
	
		patch = img[row:row + rfSize, col:col + rfSize] # 6 * 6 * 3

		b = patch.flatten() #chaque ligne représente un patch

		patches[i] = b
	
	return np.vstack(patches) #on transforme chaque patch un nparray

	
def extract_features(X, centroids, rfSize, CIFAR_DIM):
	
	n_centroids = centroids.shape[0]
	newX = np.zeros((X.shape[0], n_centroids*4)); #narray à return à la fin avec les images reformées égale à 4K
	progress = progressbar.ProgressBar(widgets=['Extracting features',progressbar.Bar(), ' ', progressbar.Percentage(), ' '])
	for i in progress(range(X.shape[0])):

		progress.update(i)
		
		#extraction convolutionelle en plusieurs étapes
		
		#on sépare les couleurs
		r = X[i,:1024]
		g = X[i,1024:2048]
		b = X[i,2048:]
		
		rs = np.reshape(r, (32,32))
		gs = np.reshape(g, (32,32))
		bs = np.reshape(b, (32,32))
		
		
		#on reshape une image en colonne 
		rse = image.extract_patches_2d( rs, (rfSize,rfSize)) #729 * 6 * 6
		gse = image.extract_patches_2d( gs, (rfSize,rfSize))
		bse = image.extract_patches_2d( bs, (rfSize,rfSize))
		
		rse = np.reshape(rse, ((len(rse), (rfSize * rfSize)))) # 729 * 36
		gse = np.reshape(gse, ((len(gse), (rfSize * rfSize))))
		bse = np.reshape(bse, ((len(bse), (rfSize * rfSize))))
		
		patches = np.hstack((rse,gse,bse)) #chaque caractéristique est reformée en un grand vecteur de dimension 729 * 108 (729 blocs)
		
		patches = normalize(patches)
		
		xx = np.sum(patches ** 2, 1, keepdims=True) #729 vecteurs colonnes
		cc = np.sum(centroids ** 2, 1, keepdims=True).T #1600 vecteurs en colonne 
		xc = np.dot(patches, centroids.T) #matrice n_centroids * 729
		
		z = np.sqrt(cc + (xx - 2 * xc)) #distance = xx^2 + cc^2 - 2 * xx * cc
		mu = z.mean(1, keepdims=True) #distance moyenne entre chaque bloc et les n_centroids
		patches = np.maximum(mu - z, 0) #on met 0 si la distance par rapport au centroid dépasse la moyenne
		
		prows = pcols = CIFAR_DIM[0] - rfSize + 1 # n - w + 1 = 27
		num_centroids = centroids.shape[0]
		
		patches = patches.reshape(prows, pcols, num_centroids) 
		
		halfr = int(np.rint(prows / 2))
		halfc = int(np.rint(pcols / 2))
		
		#on récupère les 4 quadrants
		q1 = np.sum(patches[0:halfr, 0:halfc, :], (0, 1))
		q2 = np.sum(patches[halfr:, 0:halfc, :], (0, 1))
		q3 = np.sum(patches[0:halfr, halfc:, :], (0, 1))
		q4 = np.sum(patches[halfr:, halfc:, :], (0, 1))
		
		#on reforme les caractéristiques en concaténant les 4 quadrants
		newX[i] = np.hstack((q1.flatten(), q2.flatten(), q3.flatten(), q4.flatten()))
		
	#return(n_images , 4 * K)
	return np.vstack(newX)
        
	
if __name__ == '__main__':
	
	(X_train,y_train), (X_test, y_test) = load_data()
	
	print "Extraction des patches"
	
	patches = getPatch(numPatches, X_train,  rfSize)
	
	y_train = y_train.ravel()
	y_test = y_test.ravel()

	print "Normalisation"
	patches = normalize(patches)

	print "K-means"
	centroids = kmeans(patches, numCentroids, nb_kmeans)
	
	print "Début extraction des features train"					  
	trainX = extract_features(X_train, centroids, rfSize, CIFAR_DIM)
	
	print "Début extraction des features test"
	testX = extract_features(X_test, centroids, rfSize, CIFAR_DIM)
	
	trainX_mean = np.mean(trainX)
	print "Ecart type de trainX pour standardiser"
	trainX_st_dev = np.sqrt(np.var(trainX)+0.01)
	
	print "Soustraction de trainX par la moyenne pour standardiser"
	trainXSubMean = (trainX - trainX_mean) 
	print "Soustraction de testX par la moyenne pour standardiser"		
	testXSubMean = (testX - trainX_mean) 
	
	print "Début Standardisation train"
	trainXs = trainXSubMean / trainX_st_dev
	
	print "Début Standardisation test"
	testXs = testXSubMean / trainX_st_dev

	
	from sklearn import svm
	print "Début Classification"
	clf = svm.SVC(verbose=1, shrinking=False, C=c)
	clf.fit(trainXs, y_train) 
	

	print "Début test"
	res = clf.score(testXs, y_test)
	
	print "Accuracy test : ", res
	
	




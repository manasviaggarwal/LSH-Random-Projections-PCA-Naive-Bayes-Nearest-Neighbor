import numpy as np
import matplotlib.pyplot as plt
from numpy import array,identity,diagonal
import os
import numpy
import pandas as pd
import sys
import random
import math
#from scipy.linalg import svd
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from random import randrange
import operator
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
import warnings
from sklearn.metrics.pairwise import pairwise_distances

def knn1(testset,trainset,testlabel,trainlabel):
	#print("knn")
	predictions=[]
	for k in range(3,4):
		# print("THE VALUE OF K IS:", end=' ')
		# print(k)
		counter=0
		for i in range(len(testset)):
			label=neighbours_calculate(testset[i],k,trainset,trainlabel)
			predictions.append(label)
			if(label!=testlabel[i]):
				counter=counter+1
		print("THE ACCURACY OF OUR KNN CLASSIFIER IS:",end=" ")
		print(float((len(testset)-counter)/len(testset))*100)
	
		return predictions,(float(len(testset)-counter)/len(testset))*100
	
		#print(label,testlabel[i])

def scikit_knn(testset,trainset,testlabel,trainlabel):
	#global testset,trainset,trainlabel,testlabel
	knn_classifier = KNeighborsClassifier(n_neighbors=3)
	#print(testlabel)
	knn_classifier.fit((trainset),np.ravel(trainlabel))
	
	predictions=knn_classifier.predict(testset)
	# print(len(predictions))
	# print("+++++++++++++++++++++++++++++++++++++")
	counter=0
	for i1 in range(len(predictions)):
		if predictions[i1]!=testlabel[i1]:
			counter=counter+1 
	#print(counter)
	print("THE ACCURACY OF SKLEARN KNN CLASSIFIER IS :: %f" % (float((len(testset)-counter)/len(testset))*100))
	return predictions,(float(len(testset)-counter)/len(testset)*100)

def neighbours_calculate(testsetinstance,k,trainset,trainlabel):
	#global testset,trainset,trainlabel,testlabel
	distances=[]
	for i in range(len(trainset)):
		dist=euclidian_distance(testsetinstance,trainset[i])
		distances.append((int(trainlabel[i][0]),dist))
	distances.sort(key=operator.itemgetter(1))
	#print(distances)
	nearest_neighbours=[]
	for j in range(k):
		nearest_neighbours.append((distances[j][0]))
	#return nearest_neighbors
	labels={}
	#print(nearest_neighbours)
	for i in range(len(nearest_neighbours)):
		if nearest_neighbours[i] not in labels:
			labels[nearest_neighbours[i]]=1
		else:
			labels[nearest_neighbours[i]]=labels[nearest_neighbours[i]]+1
	v=list(labels.values())

	k=list(labels.keys())
	return max(labels, key=labels.get)



def euclidian_distance(data1,data2):
	distance=0.0
	for i in range(0,len(data1)):
		f1=float(data1[i])-float(data2[i])
		distance=distance+pow(f1,2)
	return math.sqrt(distance)

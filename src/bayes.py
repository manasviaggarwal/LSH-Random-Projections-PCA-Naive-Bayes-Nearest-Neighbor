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
import warnings
from itertools import combinations


def sklearn_bayes(testset,trainset,testlabel,trainlabel):
 #SKLEARN BAYES IMPLEMENTATION.....IT RETURNS THE PREDICTIONSA AND ACCURACY
	gauss_classifier = GaussianNB()
	gauss_classifier = gauss_classifier.fit((trainset),np.ravel(trainlabel))
	
	predictions=gauss_classifier.predict(testset)
	counter=0
	for i1 in range(len(predictions)):
		if predictions[i1]!=testlabel[i1]:
			counter=counter+1 
	#print(counter)
	print("THE ACCURACY OF SKLEARN BAYES CLASSIFIER IS :%f" % (float((len(testset)-counter)/len(testset))*100))
	return predictions,(float(len(testset)-counter)/len(testset)*100)

	

def calc_mean_stddev(train,dic):
	#CALCULATES MEAN AND STANDARD DEVIATION
	mean_dictionary={}
	standard_deviation={}
	sum_tot=0
	#print(len(dic))
	for key in dic:
		A=dic[key]
		A=np.array(A)
		mean_dictionary[key]=[]
		for i in range(len(A[0])):
			sum_tot=0
			for j in range(len(A)):
				sum_tot=sum_tot+A[j][i]
			sum_tot=sum_tot/(len(A))
			mean_dictionary[key].append(sum_tot)	
	for key in dic:
		A=dic[key]
		A=np.array(A)
		standard_deviation[key]=[]
		diff=[0 for i in range(len(A[0]))]		
		for i in A:
			diff=diff+pow((i-mean_dictionary[key]),2)
		#diff=math.sqrt(diff/(len(A)))
		diff=map(math.sqrt,diff/len(A))
		#D=D/(len(A))
		#dic2[key]=list(diff/len(A))
		standard_deviation[key]=list(diff)
	return mean_dictionary,standard_deviation


def priors(trainset,dic):
	#CALCULATES ALL THE PRIORS
	prior={}
	for key in dic:
		A=dic[key]
		prior[key]=(len(A)/len(trainset))
	return prior

def predictions_calculate(testset,testlabel,mean_dictionary,standard_deviation,prior):
	#prob_calc(train[1])
	counter=0
	predictions=[]
	#print(prior)
	for i in range(len(testset)):
		i1=calculateClassProbabilities(testset[i],mean_dictionary,standard_deviation,prior)
		predictions.append(i1)
		i2=testlabel[i]
		if i1!=i2[0]:
			#print("h")
			counter=counter+1
	
	print("THE ACCURACY OF OUR BAYES CLASSIFIER IS:",end=" ")
	print(((len(testset)-counter)/len(testset))*100)
	
	return predictions,((float(len(testset)-counter)/len(testset))*100)

def joint_log_likelihood1(X):
        #check_is_fitted(self, "classes_")
        print(len(dic1[0]))
        sigma_=dic1.reshape(len(dic1),len(dic1[0]))
        theta_=dic1.reshape(len(dic1),len(dic1[0]))

        X=np.asarray(X)
        #X = check_array(X)
        joint_log_likelihood = []
        for i in range(len(dic1)):
            jointi = np.log(prior[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - theta_[i, :]) ** 2) /
                                 (sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        print(joint_log_likelihood)

def bayes_twitter(labels,label_dict,priors,testset,testlabel):
	sums={}
	predictions=[]
	for key in (label_dict):
		dic=label_dict[key]
		if key not in sums:
			sums[key]=0
		sums[key]=sum(dic.values())
	for i in range(len(testset)):
		maxi=0
		class_prob={}
		for key in label_dict:
			dic=label_dict[key]
			#print(dic)
			if key not in class_prob:
				class_prob[key]=priors[key]
			for word in (str(testset[i][0])).split():
				if(word in dic):
					class_prob[key]=class_prob[key]*((dic[word]+1)/(sums[key]+2998))
					#print(class_prob)
				else:
					class_prob[key]=class_prob[key]*((1)/(sums[key]+2998))
		#print(class_prob)
		a=max(class_prob.items(),key=operator.itemgetter(1))[0]
		predictions.append(a)

	accuracy=0
	count=0
	for i in range(len(predictions)):
	    if(predictions[i]!=testlabel[i][0]):
	            count+=1
	accuracy=((len(predictions)-count)/len(predictions))*100

	return(accuracy,predictions)

def calculateClassProbabilities(inp,mean_dictionary,standard_deviation,prior):
	#global dic1,dic2
	#CALCULATES THE PROBABILITIES
	class_cond_prob = {}
	#print(dic2[0][30])
	for key in mean_dictionary:
		class_cond_prob[key]=0
		#print(len(dic1[key]))
		for k in range(len(mean_dictionary[key])):
			mean=mean_dictionary[key][k]
			standarddev=standard_deviation[key][k]
			inp1 = inp[k]
			if standarddev!=0:
				try:
				#print(x,mean,standarddev)
					#if(class_cond_prob[key]<1e123):
					if(abs(class_cond_prob[key])!=float('inf')):
						A=class_cond_prob[key]
						class_cond_prob[key] += prob(standarddev,mean,inp1)
				except:
						class_cond_prob[key]=A
		class_cond_prob[key]=class_cond_prob[key]+np.log(prior[key])
		#print(class_cond_prob)
	#print(class_cond_prob)
	return max(class_cond_prob, key=class_cond_prob.get)




def prob(stdandard_dev,mean,inp):
	# exponent = math.exp(-(math.pow(inp-mean,2)/(2*math.pow(stdandard_dev,2))))
	# return (1 / (math.sqrt(2*math.pi) * stdandard_dev)) * exponent
	var1=(math.pow(stdandard_dev,2))
	var1=2*var1
	var2=math.pow(inp-mean,2)
	pow1 = (-(var2/var1))
	return np.log((1 / (math.sqrt(2*math.pi) * stdandard_dev))) +pow1


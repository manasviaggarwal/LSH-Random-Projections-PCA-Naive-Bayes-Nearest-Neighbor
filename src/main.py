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
import re, string, unicodedata
import nltk
import warnings
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
import bayes as b1
import nn as nn1
import projections as pro1
import lsh as lsh1

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.nan)

prior={}
dic1={}
dic2={}
testset=[]
trainset=[]
trainlabel=[]
testlabel=[]
prior={}	
train=[]
labels=[]
	

def F1_score_1(testlabel,predictions):
	
	for i in range(len(testlabel)):
	
		false_negative=0
		false_positive=0
		true_negative=0
		true_positive=0
		if testlabel[i]!=predictions[i]:
			if predictions[i]==0:
				false_negative=false_negative+1
			else:
				false_positive=false_positive+1
		else:
			if predictions[i]==0:
				true_negative=true_negative+1
			else:
				true_positive=true_positive+1
		precision=0
		recall=0
		precision=true_positive/(true_positive+false_positive)
		recall=true_positive/(true_positive+false_negative)
		f1_score_micro=0
		f1_score_macro=0


def F1_score(testlabel,predictions):
		return ((f1_score(testlabel, predictions, average='macro')),(f1_score(testlabel, predictions, average='micro')))	


def cross_validation_k(train,labels,k):
	k=10
	#global train,labels
	classes={}
	index=0
	for labelinst in labels:
		#print(labelinst)
		if labelinst[0] in classes:
			classes[labelinst[0]].add(index)
		else:
			classes[labelinst[0]] = {index}
		index=index+1
	fold_classes_list={}
	for label in classes:

	    l=len(list(classes[label]))
	    dataset_copy=list(classes[label])
	    dataset_split = list()
	    fold_size = (int(l / k))
	    for i in range(k):
	        fold = list()
	        while len(fold) < fold_size:
	            index = randrange(len(dataset_copy))
	            fold.append(dataset_copy.pop(index))
	        dataset_split.append(fold)
	    #print(dataset_split)
	    fold_classes_list[label]=dataset_split
	#print(fold_classes_list[0])
	list_k_fold=[0 for i in range(k)]
	list_k_fold1=[0 for i in range(k)]
	
	for i in range(k):
		list_small=[]
		for label in fold_classes_list:
			list_small.extend(fold_classes_list[label][i])			
		list_k_fold[i]=list(list_small)
	#print(list_k_fold)
	
	return list_k_fold
		





def testing_dolphin_pubmed(testfile,labelfile):
	print("FOR DOLPHIN DATA SET: ")
	#f2.write(str("DOLPHIN DATA SET"))
	train = pd.read_csv(testfile,delimiter=' ',header=None)
	labels=pd.read_csv(labelfile,delimiter=' ',header=None)
	train=np.array(train)
	labels=np.array(labels)
	predictions,acc=predictions_calculate(train,labels)
	predictions,acc=priors(dic)
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)

	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)

	predictions,acc=knn1()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)

	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)

	predictions,acc=scikit_knn()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)
	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)


	predictions,acc=sklearn_bayes()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)
	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)


def testing_twitter(testfile,labelfile):
	print("FOR TWITTER DATA SET: ")
	train=[]	
	labels=[]
	train = pd.read_csv(testfile,header=None)
	labels=pd.read_csv(labelfile,header=None)
	bag_of_words = set()
	finalmat = []
	words1=set()
	for i,sentence in train.iterrows():
		text = {}
		
		for word in sentence[0].strip().split():
			if word not in stopwords.words('english'): 	
				if word in text:
					text[word] += 1
				else:
					text[word]=1
		
		finalmat.append(text)
		bag_of_words.update(text)	
	#print(bag_of_words)	 	
	mat = [[(text[word] if word in text else 0) for word in bag_of_words] for text in finalmat]
	train=np.array(mat)
	labels=np.array(labels)
	#calc_mean_stddev(train,dic)
	predictions,acc=predictions_calculate(train,labels)
	predictions,acc=priors(dic)
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)

	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)

	predictions,acc=knn1()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)

	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)

	predictions,acc=scikit_knn()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)
	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)


	predictions,acc=sklearn_bayes()
	print("Our bayes classifier accuracy is : ",end=" ")
	print(acc)
	a,b=F1_score(labels,predictions)
	print("Our bayes classifier F1 score (macro and micro) are : ",end=" ")
	print(a,b)

	
	# f.write("Bayes,KNN,Sklearn_knn,sklearn_bayes")
	# print("Bayes,KNN,Sklearn_knn,sklearn_bayes:")
	# f2.write(str("F1 MACRO SCORE:")+str(f1_macro_average)+str("  F1 MICRO SCORE:")+str(f1_micro_average)+str("  ACCURACY:")+str(accuracy_average))
	# print(str("F1 MACRO SCORE:")+str(f1_macro_average)+str("  F1 MICRO SCORE:")+str(f1_micro_average)+str("  ACCURACY:")+str(accuracy_average))
	# f2.write(str("\n"))




if __name__=='__main__':
	array_of_arguments=sys.argv
	testdata_path=array_of_arguments[2]
	testlabel_path=array_of_arguments[4]
	strng=array_of_arguments[6]
	
	current_directory = os.getcwd()	
	#strng="twitter"
	#strng="pubmed"
	#f2=open('task3.txt','w')
	if strng=="dolphins" or strng=="pubmed":
		# f=open("task_3.txt",'w')
		# f11=open("task_4.txt",'w')
		prior={}
		dic1={}
		dic2={}
		
		trainset=[]
		trainlabel=[]
		testset = pd.read_csv(testdata_path,delimiter=' ',header=None)
		testlabel=pd.read_csv(testlabel_path,delimiter=' ',header=None)
		testset=np.array(testset)
		testlabel=np.array(testlabel)
		prior={}
		R=[]
		train=[]
		labels=[]
		if strng=="dolphins":
			print("FOR DOLPHIN DATA SET: ")
			final_directory = os.path.join(current_directory, r'dolphins_D_matrices')
			if not os.path.exists(final_directory):
				os.makedirs(final_directory)
			final_directory_out = os.path.join(current_directory, r'output_plots')
			if not os.path.exists(final_directory_out):
				os.makedirs(final_directory_out)
			f=open("task_3_dolphin.txt",'w')
			f11=open("task_4_dolphin.txt",'w')
			f.write("********************DOLPHIN DATASET***************************************")
			#f2.write(str("DOLPHIN DATA SET\n"))
			f11.write("********************DOLPHIN DATASET***************************************")
			train = pd.read_csv("dolphins.csv",delimiter=' ',header=None)
			labels=pd.read_csv("dolphins_label.csv",delimiter=' ',header=None)
			#f=open(final_directory+"/Accuracy.txt",'w')
		else:
			print("FOR PUBMED DATA SET:\n ")
			f=open("task_3_pubmed.txt",'w')
			f11=open("task_4_pubmed.txt",'w')
			final_directory = os.path.join(current_directory, r'pubmed_D_matrices')
			if not os.path.exists(final_directory):
				os.makedirs(final_directory)
			final_directory_out = os.path.join(current_directory, r'output_plots')
			if not os.path.exists(final_directory_out):
				os.makedirs(final_directory_out)
			f.write("********************PUBMED DATASET***************************************")
			f11.write("********************PUBMED DATASET***************************************")
			#f2.write(str("DOLPHIN DATA SET\n"))
			#f=open(final_directory+"/Accuracy.txt",'w')
			
			
			train = pd.read_csv("pubmed.csv",delimiter=' ',header=None)
			labels=pd.read_csv("pubmed_label.csv",delimiter=' ',header=None)
		
		trainset=np.array(train)
		trainlabel=np.array(labels)

		train_pca=trainset
		train_labels=trainlabel

		predictions=[]
		accuracy_average=[0 for i in range(4)]
		f1_macro_average=[0 for i in range(4)]
		f1_micro_average=[0 for i in range(4)]

		f1_micro_average_list=[0 for i in range(4)]
		f1_macro_average_list=[0 for i in range(4)]
		accuracy_average_list=[0 for i in range(4)]
		for i in range(4):
			f1_micro_average_list[i]=[]
			f1_macro_average_list[i]=[]
			accuracy_average_list[i]=[]
		
		n,m=trainset.shape
		#print(n,m)
		j21=2
		D=[]

		print("PERFORMING OPERATIONS ON R MATRIX")
		while j21<=int(m/2):
			dic={}
			accuracy_average=[0 for i in range(4)]
			f1_macro_average=[0 for i in range(4)]
			f1_micro_average=[0 for i in range(4)]
			print("THE VALUE OF D IS:",end=' ')
			print(j21)
			D.append(j21)
			f.write("THE VALUE OF D IS:")
			f.write(str(j21))
			f.write(str("\n\n"))
			f11.write("THE VALUE OF D IS:")
			f11.write(str(j21))
			f11.write(str("\n\n"))
			train_R=pro1.random_projection(m,n,trainset,j21,final_directory)
			test_R=pro1.random_projection(m,n,testset,j21,final_directory)
			j21=j21*2		
			for i1 in range(len(train_R)):
				if trainlabel[i1][0] not in dic:
					dic[trainlabel[i1][0]]=[]
				dic[trainlabel[i1][0]].append(train_R[i1])								
			dic1,dic2=b1.calc_mean_stddev(train_R,dic)
			prior=b1.priors(train_R,dic)
			#print(trainset.shape)
			predictions,acc=b1.predictions_calculate(test_R,testlabel,dic1,dic2,prior)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[0]+=b
			f1_macro_average[0]+=a
			accuracy_average[0]+=acc

			predictions,acc=nn1.knn1(test_R,train_R,testlabel,trainlabel)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[1]+=b
			f1_macro_average[1]+=a
			accuracy_average[1]+=acc

			predictions,acc=nn1.scikit_knn(test_R,train_R,testlabel,trainlabel)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[2]+=b
			f1_macro_average[2]+=a
			accuracy_average[2]+=acc

			predictions,acc=b1.sklearn_bayes(test_R,train_R,testlabel,trainlabel)
			F1_score(testlabel,predictions)
			f1_micro_average[3]+=b
			f1_macro_average[3]+=a
			accuracy_average[3]+=acc
		
			for i in range(4):
				f1_micro_average_list[i].append(f1_micro_average[i])
				f1_macro_average_list[i].append(f1_macro_average[i])
				accuracy_average_list[i].append(accuracy_average[i])
			f.write("F1 MACRO SCORE:\n")
			f11.write("F1 MACRO SCORE:\n")
			f.write("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n"))
			f11.write(str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
			f.write("F1 MICRO SCORE:\n")
			f11.write("F1 MICRO SCORE:\n")
			f.write("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n"))
			f11.write("Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
			f.write("ACCURACY:\n")
			f11.write("ACCURACY:\n")
			f.write("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n"))
			f11.write("Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
			print("F1 MACRO SCORE:\n")
			print("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
			print("F1 MICRO SCORE:\n")
			print("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
			print("ACCURACY:\n")
			print("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n")+"Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
	
		legends=["CustomBayes","CustomKNN","Sklearn_Bayes","Sklearn_KNN"]
		if strng=="dolphins":
			for i in range(4):
				#print(f1_micro_average_list[i])

				plt.plot(D,f1_macro_average_list[i])
				
			plt.xlabel('D Values')
			plt.ylabel('F1-Macro-Score')	
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4_dolphin_f1-Macro-accuracy.png')
			plt.clf()
			for i in range(4):
				plt.plot(D,f1_micro_average_list[i])
			plt.xlabel('D Values')
			plt.ylabel('F1-Micro-Score')
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4_dolphin_f1-Micro-accuracy.png')
			plt.clf()
			for i in range(4):
				plt.plot(D,accuracy_average_list[i])
			plt.xlabel('D Values')
			plt.ylabel('Accuracy')
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4_dolphin_Accuracy.png')
			plt.clf()
		elif strng=="pubmed":
			for i in range(4):
			#print(f1_micro_average_list[i])

				plt.plot(D,f1_macro_average_list[i])
			
			plt.xlabel('D Values')
			plt.ylabel('F1-Macro-Score')	
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4__pubmed_f1-Macro-accuracy.png')
			plt.clf()
			for i in range(4):
				plt.plot(D,f1_micro_average_list[i])
			plt.xlabel('D Values')
			plt.ylabel('F1-Micro-Score')
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4_pubmed_f1-Micro-accuracy.png')
			plt.clf()
			for i in range(4):
				plt.plot(D,accuracy_average_list[i])
			plt.xlabel('D Values')
			plt.ylabel('Accuracy')
			plt.legend(legends)	
			#plt.show()
			plt.savefig(final_directory_out+'/task_4_pubmed_Accuracy.png')
			plt.clf()

		f1_micro_average_list=[]
		f1_macro_average_list=[]
		accuracy_average_list=[]

		print("PERFORMING OPERATIONS ON ORIGINAL MATRIX")
		f.write("\n**************PERFORMING OPERATIONS ON ORIGINAL MATRIX*****************\n")
		f11.write("\n*************PERFORMING OPERATIONS ON ORIGINAL MATRIX******************\n")
		
		
		accuracy_average=[0 for i in range(4)]
		f1_macro_average=[0 for i in range(4)]
		f1_micro_average=[0 for i in range(4)]

		f1_micro_average_list=[0 for i in range(4)]
		f1_macro_average_list=[0 for i in range(4)]
		accuracy_average_list=[0 for i in range(4)]

		prior={}
		dic1={}
		dic2={}
		
		dic={}
		
		label1=[]
		#print(list_k_fold)
		
		for i1 in range(len(trainset)):
			if trainlabel[i1][0] not in dic:
				dic[trainlabel[i1][0]]=[]
			dic[trainlabel[i1][0]].append(trainset[i1])				
		
		dic1,dic2=b1.calc_mean_stddev(trainset,dic)
		prior=b1.priors(trainset,dic)
		predictions,acc=b1.predictions_calculate(testset,testlabel,dic1,dic2,prior)
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[0]+=b
		f1_macro_average[0]+=a
		accuracy_average[0]+=acc

		predictions=[]
		predictions,acc=nn1.knn1(testset,trainset,testlabel,trainlabel)
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[1]+=b
		f1_macro_average[1]+=a
		accuracy_average[1]+=acc

		predictions=[]
		predictions,acc=nn1.scikit_knn(testset,trainset,testlabel,trainlabel)
		# print(len(predictions))
		# print(len(testlabel))
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[2]+=b
		f1_macro_average[2]+=a
		accuracy_average[2]+=acc

		predictions=[]
		predictions,acc=b1.sklearn_bayes(testset,trainset,testlabel,trainlabel)
		F1_score(testlabel,predictions)
		f1_micro_average[3]+=b
		f1_macro_average[3]+=a
		accuracy_average[3]+=acc
		f.write("\nF1 MACRO SCORE:\n")
		f11.write("\nF1 MACRO SCORE:\n")
		f.write("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n"))
		f11.write(str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
		f.write("\nF1 MICRO SCORE:\n")
		f11.write("\nF1 MICRO SCORE:\n")
		f.write("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n"))
		f11.write("Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
		f.write("\nACCURACY:\n")
		f11.write("\nACCURACY:\n")
		f.write("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n"))
		f11.write("Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
		
		
		print("F1 MACRO SCORE:\n")
		print("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
		print("F1 MICRO SCORE:\n")
		print("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
		print("ACCURACY:\n")
		print("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n")+"Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
		
		# print(str("F1 MACRO SCORE:")+str(f1_macro_average)+str("\n")+str("F1 MICRO SCORE:")+str(f1_micro_average)+str("\n")+str("ACCURACY:")+str(accuracy_average)+str("\n"))		
		# print("\n")
		
		if strng=="dolphins":
			lsh1.LSH_main(train_pca,train_labels,testset,testlabel,"dolphin")
		else:
			lsh1.LSH_main(train_pca,train_labels,testset,testlabel,"pubmed")
		f.close()
		f11.close()

		
			
	elif strng=="twitter":
		#print("hello")
		testset = pd.read_csv(testdata_path,header=None)
		testlabel=pd.read_csv(testlabel_path,header=None)
		
		accuracy_average=[0 for i in range(4)]
		f1_macro_average=[0 for i in range(4)]
		f1_micro_average=[0 for i in range(4)]
		print("FOR TWITTER DATA SET: ")
		final_directory = os.path.join(current_directory, r'twitter_D_matrices')
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
		#f=open(final_directory+"/Accuracy.txt",'w')
		final_directory_out = os.path.join(current_directory, r'output_plots')
		if not os.path.exists(final_directory_out):
			os.makedirs(final_directory_out)
		f=open("task_3_twitter.txt",'w')
		f11=open("task_4_twitter.txt",'w')
			#f2.write(str("DOLPHIN DATA SET\n"))
			
		dic1={}
		dic2={}
		
		trainset=[]
		trainlabel=[]
		
		prior={}
		word_freq={}
		train=[]	
		labels=[]
		trainset = pd.read_csv("twitter.txt",header=None)
		trainlabel=pd.read_csv("twitter_label.txt",header=None)
		f1_micro_average_list=[0 for i in range(4)]
		f1_macro_average_list=[0 for i in range(4)]
		accuracy_average_list=[0 for i in range(4)]
		for i in range(4):
			f1_micro_average_list[i]=[]
			f1_macro_average_list[i]=[]
			accuracy_average_list[i]=[]

		bag_of_words = set()
		finalmat = []
		words1=set()
		text1={}
		for i,sentence in trainset.iterrows():
			text = {}
			
			for word in sentence[0].strip().split():
				if word not in stopwords.words('english'): 	
					if word in text:
						text[word] += 1
					else:
						text[word]=1
				
			finalmat.append(text)
			bag_of_words.update(text)	
		
		
		 	
		mat = [[(text[word] if word in text else 0) for word in bag_of_words] for text in finalmat]
		mat1=mat

		bag_of_words = set()
		finalmat = []
		words1=set()
		text1={}
		for i,sentence in testset.iterrows():
			text = {}
			
			for word in sentence[0].strip().split():
				if word not in stopwords.words('english'): 	
					if word in text:
						text[word] += 1
					else:
						text[word]=1
				
			finalmat.append(text)
			bag_of_words.update(text)	
		
		
		 	
		mat_test = [[(text[word] if word in text else 0) for word in bag_of_words] for text in finalmat]
		testset_binary=mat_test



		trainset_binary=np.array(mat)	
		trainset_binary_lsh=trainset_binary

		trainset=np.array(trainset)
		trainlabel=np.array(trainlabel)
		trainlabel_lsh=trainlabel

		testset=np.array(testset)
		testlabel=np.array(testlabel)

		testset_binary_lsh=np.array(testset_binary)
		new_words = []
		
		n,m=trainset_binary.shape

		# testset=train
		# testlabel=labels
		k3=10
		j21=2
		#print(m,n)
		D=[]
		f.write("\n******************************TWITTER DATASET:****************************************\n")
		f11.write("\n******************************TWITTER DATASET:****************************************\n")
		print("PERFORMING OPERATIONS ON R MATRIX")
		f.write("PERFORMING OPERATIONS ON R MATRIX\n")
		while j21<=int(m/2):
			accuracy_average=[0 for i in range(4)]
			f1_macro_average=[0 for i in range(4)]
			f1_micro_average=[0 for i in range(4)]
			print("THE VALUE OF D IS:",end=' ')
			print(j21)
			f.write("THE VALUE OF D IS:")
			f.write(str(j21))
			f.write(str("\n\n"))
			f11.write("THE VALUE OF D IS:")
			f11.write(str(j21))
			f11.write(str("\n\n"))
			D.append(j21)
			train_R=pro1.random_projection(m,n,trainset_binary,j21,final_directory)
			test_R=pro1.random_projection(m,n,testset_binary,j21,final_directory)
			#list_k_fold=cross_validation_k(train_R,labels,k3)
			j21=j21*2
			
			prior={}
			
			
			dic={}
			
			label1=[]
			
			for i1 in range(len(train_R)):
				if trainlabel[i1][0] not in dic:
					dic[trainlabel[i1][0]]=[]
				dic[trainlabel[i1][0]].append(train_R[i1])	
			
				
			dic1,dic2=b1.calc_mean_stddev(train_R,dic)
			prior=b1.priors(train_R,dic)
			predictions,acc=b1.predictions_calculate(test_R,testlabel,dic1,dic2,prior)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[0]+=b
			f1_macro_average[0]+=a
			accuracy_average[0]+=acc

			predictions,acc=nn1.knn1(test_R,train_R,testlabel,trainlabel)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[1]+=b
			f1_macro_average[1]+=a
			accuracy_average[1]+=acc

			predictions,acc=nn1.scikit_knn(test_R,train_R,testlabel,trainlabel)
			a,b=F1_score(testlabel,predictions)
			f1_micro_average[2]+=b
			f1_macro_average[2]+=a
			accuracy_average[2]+=acc

			predictions,acc=b1.sklearn_bayes(test_R,train_R,testlabel,trainlabel)
			F1_score(testlabel,predictions)
			f1_micro_average[3]+=b
			f1_macro_average[3]+=a
			accuracy_average[3]+=acc


			f1_micro_average_list.append(f1_micro_average)
			f1_macro_average_list.append(f1_macro_average)
			accuracy_average_list.append(accuracy_average)
			f.write("\nF1 MACRO SCORE:\n")
			f11.write("\nF1 MACRO SCORE:\n")
			f.write("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n"))
			f11.write(str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
			f.write("\nF1 MICRO SCORE:\n")
			f11.write("\nF1 MICRO SCORE:\n")
			f.write("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n"))
			f11.write("Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
			f.write("\nACCURACY:\n")
			f11.write("\nACCURACY:\n")
			f.write("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n"))
			f11.write("Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
			print("F1 MACRO SCORE:\n")
			print("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
			print("F1 MICRO SCORE:\n")
			print("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
			print("ACCURACY:\n")
			print("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n")+"Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
			

			

		legends=["CustomBayes","CustomKNN","Sklearn_Bayes","Sklearn_KNN"]
		for i in range(4):
			plt.plot(D,f1_micro_average_list[i])
		plt.xlabel('D Values')
		plt.ylabel('F1-MICRO-Score')	
		plt.legend(legends)	
		#plt.show()

		plt.savefig(final_directory_out+'/task_4_twitter_f1-Macro-accuracy.png')
		plt.clf()

		for i in range(4):
			plt.plot(D,f1_macro_average_list[i])
		plt.xlabel('D Values')
		plt.ylabel('F1-MACRO-Score')
		plt.legend(legends)	
		#plt.show()
		plt.savefig(final_directory_out+'/task_4_twitter_f1-Micro-accuracy.png')
		plt.clf()
		for i in range(4):
			plt.plot(D,accuracy_average_list[i])
		plt.xlabel('D Values')
		plt.ylabel('ACCURACY')
		plt.legend(legends)	
		#plt.show()
		plt.savefig(final_directory_out+'/task_4_twitter_Accuracy.png')
		plt.clf()
		k3=5

		print("PERFORMING OPERATIONS ON ORIGINAL MATRIX:\n")
		f.write("PERFORMING OPERATIONS ON ORIGINAL MATRIX:\n")
		#list_k_fold=cross_validation_k(train,labels,10)
		
		accuracy_average=[0 for i in range(4)]
		f1_macro_average=[0 for i in range(4)]
		f1_micro_average=[0 for i in range(4)]

		f1_micro_average_list=[0 for i in range(4)]
		f1_macro_average_list=[0 for i in range(4)]
		accuracy_average_list=[0 for i in range(4)]	
		prior={}
		dic1={}
		dic2={}
		
		dic={}
		
		label1=[]
		#print(list_k_fold)
		
		for i1 in range(len(trainset)):
			if trainlabel[i1][0] not in dic:
				dic[trainlabel[i1][0]]=[]
			dic[trainlabel[i1][0]].append(trainset[i1])				
		
		label_dict={}
		trainset=pd.DataFrame(data=trainset)
		#print(len(trainset))
		text={}
		for i,sentence in trainset.iterrows():    #dictionary of all words to further remove words with frequency less than 1 and stopwords as well		
			for word in sentence[0].strip().split():
				if word not in stopwords.words('english'): 	
					if word in text:
						text[word] += 1
					else:
						text[word]=1
		text1={k:val for k,val in text.items() if val>=1}
		for i,sentence in trainset.iterrows():

			for word in sentence[0].strip().split():
				if word not in stopwords.words('english') and word in text1: 
					#print("h")
					if trainlabel[i][0] not in label_dict:
						label_dict[trainlabel[i][0]]={}
					#if labels[i][0]==1:
					if word not in label_dict[trainlabel[i][0]]:
						#print()
						label_dict[trainlabel[i][0]][word]=0   	
					label_dict[trainlabel[i][0]][word]+=1

		prior=b1.priors(trainset,dic)		
		predictions,acc=b1.bayes_twitter(trainlabel,label_dict,prior,testset,testlabel)
		# calc_mean_stddev(dic)
		# priors(dic)
		#predictions,acc=predictions_calculate(testset,testlabel)
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[0]+=b
		f1_macro_average[0]+=a
		accuracy_average[0]+=acc
		predictions=[]

		predictions,acc=nn1.knn1(testset_binary,trainset_binary,testlabel,trainlabel)
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[1]+=b
		f1_macro_average[1]+=a
		accuracy_average[1]+=acc

		predictions=[]
		predictions,acc=nn1.scikit_knn(testset_binary,trainset_binary,testlabel,trainlabel)
		# print(len(predictions))
		# print(len(testlabel))
		a,b=F1_score(testlabel,predictions)
		f1_micro_average[2]+=b
		f1_macro_average[2]+=a
		accuracy_average[2]+=acc

		predictions=[]
		predictions,acc=b1.sklearn_bayes(testset_binary,trainset_binary,testlabel,trainlabel)
		F1_score(testlabel,predictions)
		f1_micro_average[3]+=b
		f1_macro_average[3]+=a
		accuracy_average[3]+=acc

		f.write("\nF1 MACRO SCORE:\n")
		f11.write("\nF1 MACRO SCORE:\n")
		f.write("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n"))
		f11.write(str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
		f.write("\nF1 MICRO SCORE:\n")
		f11.write("\nF1 MICRO SCORE:\n")
		f.write("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n"))
		f11.write("Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
		f.write("\nACCURACY:\n")
		f11.write("\nACCURACY:\n")
		f.write("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n"))
		f11.write("Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
		print("F1 MACRO SCORE:\n")
		print("Bayes: "+str(f1_macro_average[0])+str("\n")+"KNN: "+str(f1_macro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_macro_average[2])+"\nsklearn_bayes: "+str(f1_macro_average[3])+"\n")
		print("F1 MICRO SCORE:\n")
		print("Bayes: "+str(f1_micro_average[0])+str("\n")+"KNN: "+str(f1_micro_average[1])+str("\n")+"Sklearn_knn: "+str(f1_micro_average[2])+"\nsklearn_bayes: "+str(f1_micro_average[3])+"\n")
		print("ACCURACY:\n")
		print("Bayes: "+str(accuracy_average[0])+str("\n")+"KNN: "+str(accuracy_average[1])+str("\n")+"Sklearn_knn: "+str(accuracy_average[2])+"\nsklearn_bayes: "+str(accuracy_average[3])+"\n")
		
		lsh1.LSH_main(trainset_binary_lsh,trainlabel,testset_binary_lsh,testlabel,"twitter")
		f.close()
		f11.close()








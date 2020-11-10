from copy import copy
from itertools import combinations
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import operator
#import prog1
import pandas as pd
from nltk.corpus import stopwords
from random import randrange
#from prog1.py import cross_validation_k,scikit_knn
from sklearn.neighbors import KNeighborsClassifier
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
def lsh_new(training_set,dim):
	#def lsh_ac(training_set,dim,i):
    curr=os.getcwd()
    final=os.path.join(curr,r'LSH_files')
    if not os.path.exists(final):
        os.makedirs(final)

    np.random.seed(42)
    row,col = training_set.shape
    training_set=np.array(training_set)
    training_set=training_set-training_set.mean(axis=1,keepdims=True)
    training_set[training_set<=0]=0
    training_set[training_set>0]=1
    training_set=np.array(training_set,dtype=int)
    returning_set=pd.DataFrame(np.zeros((row,dim)))
    for i in range(dim):
        prn = np.random.permutation(col)+1
        new_x=training_set*prn
        new_x[new_x==0] = col+1
        returning_set[i]=np.array(new_x.min(axis=1),dtype=int)
    #f=open(final,'/task_6_dataset'+str(i)+'dimension'+str(dim)+'.txt','w')
   # f.write(str(np.array(returning_set)))
   # f.close()
    return np.array((returning_set))

def LHS_classifier(m,n,train):
	#  lsh = LocalitySensitiveHashing( 
 #                   datafile = "dolphins.csv",
 #                   dim = m,
 #                   r = 50,            
 #                   b = 100,              
 #           )
	#  lsh.get_data_from_csv()
	#  lsh.initialize_hash_store()
	#  lsh.hash_all_data()
	# # similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()
	# # print(similarity_neighborhoods)
	#  show_data_for_lsh()
	lsh_model = LSH(train)
	num_of_random_vectors = 15
	lsh_model.train(num_of_random_vectors)

	#find the 5 nearest neighbors of data[1] while searching in 10 buckets 
	lsh_model.query(data[1,:], 5, 10)
	

def Jacaard_distance():
	a1,b1=set(c1),set(c2)
	return (1-len(a&b)/len(a|b)) if len(a|b) else float('inf')


def PCA_dimension_reduction(train,comp):
	pca = PCA(n_components=comp)
	principalComponents = pca.fit_transform(train)
	principalDf = pd.DataFrame(principalComponents)
	#print(np.array(principalDf))
	return np.array(principalDf)



def dimension_reduction_LSH(train,dimensions):
	m,n=np.array(train).shape
	trnsmat=np.random.standard_normal(n*dimensions)
	trnsmat=trnsmat.reshape(n,dimensions)
	permutaion_matrix = np.asmatrix(trnsmat)
	return permutaion_matrix


def multiply_proj_trainset(train,permutation_matrix1):
	# print(np.array(train).shape)
	# print(np.array(permutaion_matrix1).shape)
	V_X=np.array(train).dot(np.array(permutation_matrix1))
	return V_X

def hash_index(train,hash_tables):
	Hashtable={}
	dimensions=len(train[0])
	#print(V_X[0])
	index_full=[]	
	permutation_matrix={}
	for j in range(hash_tables):
		trnsmat=np.random.standard_normal(1*dimensions)
		trnsmat=trnsmat.reshape(1,dimensions)
		permutation_matrix1 = np.asmatrix(trnsmat)
		# permutation_matrix1=dimension_reduction_LSH(train,dimensions)
		permutation_matrix[j]=np.array(permutation_matrix1).T
		#print(len(permutation_matrix[j]))
		V_X=multiply_proj_trainset(train,np.array(permutation_matrix1).T)
		#V_X=multiply_proj_trainset(train,per[j])
		#print(V_X)
		#break
		hashtable_iter={}
		#for k1 in range(len(V_X)):			
		U=np.random.uniform(0,5)
		#print(U)	
		index_full.append(U)
		#print(len(V_X))
		for i in range(len(V_X)):
			#L=math.sqrt(V_X[i].dot(V_X[i]))
			#print(L)
			#break
			index=int((V_X[i]+U)/5)			
			if index not in hashtable_iter:
				hashtable_iter[index]=[]
			hashtable_iter[index].append(i)
			#print(len(hashtable_iter[index]))
		Hashtable[j]=hashtable_iter

	return Hashtable,index_full,permutation_matrix


def cross_validation_k(train,labels,k):
	#k=10
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


def scikit_knn(testset,trainset,testlabel,trainlabel):
	#global testset,trainset,trainlabel,testlabel

	knn_classifier = KNeighborsClassifier(n_neighbors=(3 if len(trainset)>=3 else len(trainset)))
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
	#print("THE ACCURACY OF SKLEARN KNN CLASSIFIER IS :: %f" % (float((len(testset)-counter)/len(testset))*100))
	return predictions,(float(len(testset)-counter)/len(testset)*100)
		

def Query(testset,Hashtable,permutation_matrix,index_full,hash_table):
	S=[]
	#print(Hashtable[0])
	# for key in Hashtable[0]:
	# 	print(key)
	#U_X=testset.dot(permutaion_matrix)
	s=[0 for i1 in range(len(testset))]	
	#print(permutaion_matrix[0])
	for i in range(len(testset)):
		s[i]=set()	
	idx=[]
	# for j in range(hash_table):	
	# 	for i in Hashtable[j]:
	# 		idx.append(i)
	# 	print(j)
	# 	print(idx)
	# 	idx=[]
	# ind=[0 for i1 in range(len(testset))]
	# for i in range(len(testset)):

	# 	ind[i]=[]
	for i in range(hash_table): #no of hashtables
		U_X=np.array(testset).dot(np.array(permutation_matrix[i]))
		for k1 in range(len(U_X)):
			#L=math.sqrt(U_X[k1].dot(U_X[k1]))
			index=int((U_X[k1]+index_full[i])/5)
			#ind[k1].append(index)
			#for j in Hashtable:
			if index in  Hashtable[i]:
				s[k1].update((Hashtable[i][index]))
			
	# print(s)
	# print(ind)
	return s



	# Hashtable,index_full,permutaion_matrix=hash_index(trainset,20,100)

	# Queryset=Query(testset,Hashtable,permutaion_matrix,index_full,20)
	#print(Queryset[0])
def calculate_accuracy(Queryset,trainset,testlabel,trainlabel,testset):
	count=0
	f1_micro_average=0
	f1_macro_average=0
	predictions_list=[]
	#print(len(Queryset))
	# print(len(testset))
	for i in range(len(Queryset)):
		S=[]
		c=0
		d=0
		
		dic=list(Queryset[i])
		#print(dic)
		for j in range(len(dic)):			
			S.append(trainlabel[dic[j]][0])		
		dic1={}
		for k in range(len(S)): #coontains cuont respective to each label
			if S[k] not in dic1:
				dic1[S[k]]=0
			dic1[S[k]]+=1
		
		if len(dic)==0:
			print("---------")
			#print(dic1)
			print(testlabel[i])
		else:
			dic2=[]
			dic3=[]
			for j in range(len(dic)):
				dic2.append(trainset[dic[j]])
				dic3.append(trainlabel[dic[j]])
			predictions,acc1=scikit_knn(np.array(testset[i]).reshape(1,len(testset[i])),dic2,testlabel[i],dic3)
			#print((list(predictions)))
			predictions_list.append(predictions)

			#predictions=scikit_knn(np.array(test_set[i]).reshape(1,len(test_set[i])),list2,test_label[i],list3)
			#pred.append(predictions)
			if predictions!=testlabel[i][0]:
			    count+=1

   # print("miss:")
	return (float((len(testset)-count)/len(testset)*100),np.array(predictions_list))
    

		# else:
		# 	dic2=[]   #contains all the training set candidates
		# 	dic3=[]   #contains all the training set labels
		# 	for j in range(len(dic)):
		# 		dic2.append(trainset[dic[j]])
		# 		dic3.append(trainlabel[dic[j]])
			
		# 	nearest_neighbors = DataFrame({'id': list(dic)})
		# 	nearest_neighbors['distance'] = pairwise_distances((np.array(dic2)).reshape(len(dic2),len(dic2[0])), (np.array(testset[i])).reshape(1,len(testset[i])), metric='cosine').flatten()
		# 	B=(nearest_neighbors.nsmallest(7, 'distance'))
		# 	A=[]
		# 	A=B['id']
		# 	label={}
		# 	A=np.array(A)
		# 	for i1 in range(len(A)):
		# 		if trainlabel[A[i1]][0] not in label:
		# 			label[trainlabel[A[i1]][0]]=0
		# 		label[trainlabel[A[i1]][0]]+=1
		# 	a=max(label.items(),key=operator.itemgetter(1))[0]
		# 	predictions_list.append(a)
			
		# 	if a!=testlabel[i][0]:
		# 		count+=1
			
		
	
	#return (float((len(testset)-count)/len(testset)*100),np.array(predictions_list))
	
def LSH(final_directory,train,reduced_dimensions):
	# final_directory = os.path.join(current_directory, r'LSH')
	# if not os.path.exists(final_directory):
	# 	os.makedirs(final_directory)	
    np.random.seed(42)
    n,m = train.shape
    train=np.array(train)
    train=train-train.mean(axis=1,keepdims=True)
    for i in range(len(train)):
    	for j in range(len(train[0])):
    		if(train[i][j]<0):
    			train[i][j]=0
    		else:
    			train[i][j]=1
    train=np.array(train)
    reduced_array=pd.DataFrame(np.zeros((n,reduced_dimensions)))
    for i in range(reduced_dimensions):
        permutation_matrix = np.random.permutation(m)+1
        reduced_matrix=train*permutation_matrix
        for j in range(len(reduced_matrix)):
	        reduced_matrix[j] = m+1
	        reduced_array[j]=np.array(reduced_matrix.min(axis=1),dtype=int)
	# f4=open(final_directory+'/task_6_'+str(reduced_dimensions)+'.txt','w')
	# f4.write(str(np.array(reduced_array)))
    return np.array(reduced_array) #pd.DataFrame(reduced_array)

def F1_score(testlabel,predictions):
	return ((f1_score(testlabel, predictions, average='macro')),(f1_score(testlabel, predictions, average='micro')))


def LSH_main(trainset,trainlabel,testset,testlabel,str1):

	current_directory = os.getcwd()	
	final_directory = os.path.join(current_directory, r'output_plots')
	if not os.path.exists(final_directory):
		os.makedirs(final_directory)
	#f4=open(final_directory+"/task_6.txt",'w')
	if str1=="dolphin":
		f5=open("task_7_dolphins.txt",'w')
	elif str1=="pubmed":
		f5=open("task_7_pubmed.txt",'w')
	elif str1=="twitter":
		f5=open("task_7_twitter.txt",'w')

	
	# print("FOR DOLPHIN DATA SET: ")
	# training_set = pd.read_csv("pubmed.csv",delimiter=' ',header=None)
	# labels=pd.read_csv("pubmed_label.csv",delimiter=' ',header=None)
	# training_set=np.array(training_set)
	#training_set=np.array(training_set)	
	# labels=np.array(labels)
	
	#print(n,m)
	i3=2
	D=[]
	f1_micro_average_list=[]
	f1_macro_average_list=[]
	f1_macro_average=0.0
	f1_micro_average=0.0
	acc_list=[]
	acc_avrg=0.0
	f1_micro_average_list_princ_comp=[]
	f1_macro_average_list_princ_comp=[]
	f1_macro_average_princ_comp=0.0
	f1_micro_average_princ_comp=0.0
	acc_list_princ_comp=[]
	acc_avrg_princ_comp=0.0
	n,m=trainset.shape
	#train=trainset

	#print(m)
	print("LSH AND PCA")
	while(i3<=int(m/2)):
		# f5.write("D value: ")
		# f5.write(str(i3))
		# f5.write("\n")
		D.append(i3)
		f1_macro_average=0.0
		f1_micro_average=0.0
		acc_avrg=0.0
		f1_macro_average_princ_comp=0.0
		f1_micro_average_princ_comp=0.0
		
		acc_avrg_princ_comp=0.0
		print("Accuracy and F1-Scores for D value  =  ",end=' ')
		print(i3)
		f5.write("Accuracy and F1-Scores for D value  =  ")
		f5.write(str(i3)+"\n\n")
		
		#list_k_fold=cross_validation_k(training,labels,10)
		#print("k fold done")
		
		#print(np.array(train).shape)
		#print("copying done")
		#print("lsh new into")
		train_R=lsh_new(trainset,i3)
		test_R=lsh_new(testset,i3)
		trainset_pca_matrix=PCA_dimension_reduction(trainset,i3)
		testset_pca_matrix=PCA_dimension_reduction(testset,i3)
		#print(training_pca_matrix.shape)
		#print("outofpca")
		prior={}
		dic1={}
		dic2={}
		# testset=[]
		# trainset=[]
		# trainlabel=[]
		# testlabel=[]
		label1=[]
		
		#print("lsh new out")
	
		#print("k1=",end=' ')

		#print(k1)
		prior={}
		dic1={}
		dic2={}
		
		label1=[]
		
		#print(list_k_fold)
		
		predictions_pca_pred,acc=scikit_knn(testset_pca_matrix,trainset_pca_matrix,testlabel,trainlabel)
		a,b=F1_score(testlabel,predictions_pca_pred)
		f1_macro_average_princ_comp+=a
		f1_micro_average_princ_comp+=b
		
		acc_avrg_princ_comp+=acc


		Hashtable,index_full,permutaion_matrix=hash_index(train_R,10)
		#print("out of hash_ndex function")
		#print("into of query function")
		Queryset=Query(test_R,Hashtable,permutaion_matrix,index_full,10)
		#print("out of query function")
		# print("-----------------")
		# print(len(testlabel))
		#print("out of query function")
		acc,predictions=calculate_accuracy(Queryset,train_R,testlabel,trainlabel,test_R)
		#print("out of query function")
		#print(predictions)
		acc_avrg+=acc
		testlabel1=[]
		for i in range(len(testlabel)):
			testlabel1.append(testlabel[i][0])	
		#testlabel=testlabel1
		#print((np.array(testlabel1)))
		#print((predictions))
		a,b=F1_score(list(np.array(testlabel1)),list(np.array(predictions)))
		f1_micro_average+=b
		f1_macro_average+=a
		f1_micro_average_list.append(f1_micro_average)
		f1_macro_average_list.append(f1_macro_average)
		acc_list.append(acc_avrg)
		f1_micro_average_list_princ_comp.append(f1_micro_average_princ_comp)
		f1_macro_average_list_princ_comp.append(f1_macro_average_princ_comp)
		acc_list_princ_comp.append(acc_avrg_princ_comp)
		f5.write("LSH ACCURACY IS = ")
		f5.write(str(acc_avrg))
		f5.write("LSH F1-SCORE(macro) IS = ")
		f5.write(str(f1_macro_average))
		f5.write("LSH F1-SCORE(micro) IS = ")
		f5.write(str(f1_micro_average))

		f5.write("PCA ACCURACY IS = ")
		f5.write(str(acc_avrg_princ_comp))
		f5.write("PCA F1-SCORE(macro) IS = ")
		f5.write(str(f1_macro_average_princ_comp))
		f5.write("PCA F1-SCORE(micro) IS = ")
		f5.write(str(f1_micro_average_princ_comp))

		print("LSH ACCURACY IS = "+ str(acc_avrg))
		print("LSH F1-SCORE(macro) IS = "+str(f1_macro_average))
		print("LSH F1-SCORE(micro) IS = "+str(f1_micro_average))

		print("PCA ACCURACY IS = "+str(acc_avrg_princ_comp))
		print("PCA F1-SCORE(macro) IS = "+str(f1_macro_average_princ_comp))
		print("PCA F1-SCORE(micro) IS = "+str(f1_micro_average_princ_comp))


	
	
	#for i in range(4):
	
		i3=i3*2
#print(acc_list_princ_comp)
	plt.plot(D,f1_micro_average_list,label="F1-MICRO")
	plt.plot(D,f1_micro_average_list_princ_comp,label="F1-MICRO")
	plt.xlabel('D Values')
	plt.ylabel('F1-Micro-Score')
	#plt.legend(['F1-micro score'])	
	plt.legend(['LSH','PCA'])											
	#plt.show()
	plt.savefig(final_directory+'task_7_f1-Micro-score.png')
	plt.clf()
#for i in range(4):
	plt.plot(D,f1_macro_average_list,label="F1-MACRO")
	plt.plot(D,f1_macro_average_list_princ_comp,label="F1-MICRO")
	plt.xlabel('D Values')
	plt.ylabel('F1-macro-Score')
	#plt.legend(['F1-macro score'])	
	plt.legend(['LSH','PCA'])															
	#plt.show()
	plt.savefig(final_directory+'task_7_f1-Macro-score.png')
	plt.clf()
#for i in range(4):
	plt.plot(D,acc_list)
	plt.plot(D,acc_list_princ_comp)
	plt.xlabel('D Values')
	plt.ylabel('Accuracy')
	plt.legend(['LSH','PCA'])								
	#plt.show()
	plt.savefig(final_directory+'task_7_accuracy.png')
	plt.clf()
			

# print("FOR DOLPHIN DATA SET: ")
# train = pd.read_csv("dolphins.csv",delimiter=' ',header=None)
# labels=pd.read_csv("dolphins_label.csv",delimiter=' ',header=None)
# train=np.array(train)
# labels=np.array(labels)
# LSH_main(train,labels,"dolphin")


	

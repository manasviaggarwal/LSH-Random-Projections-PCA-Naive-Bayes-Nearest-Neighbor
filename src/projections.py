import numpy as np
from numpy import array,identity,diagonal
import os
import numpy
import pandas as pd
import sys
import random
import math
from math import sqrt
from random import randrange



def random_projection(m,n,train,d,final_directory):
	fin_mat=[]
	trnsmat=np.random.standard_normal(m*d)
	trnsmat=trnsmat.reshape(m,d)
	rmat = np.asmatrix(trnsmat)
	fin_mat=np.matmul(train,rmat)
	fin_mat=(1/math.sqrt(d))*fin_mat
	f1 = open(final_directory+'/task_1_'+str(d)+'.txt', 'w')
	f1.write(str((np.array(fin_mat))))
	return np.array(fin_mat)
#	j=j*2
import numpy as np
import pandas as pd 
import os
from math import sqrt,exp
from numpy.linalg import inv,det
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


train=pd.read_csv("mnist_train.csv")
y_train=train['label']
X_train=train.drop(['label'],axis=1)
X_train=np.array(X_train)
y_train=np.array(y_train)
y_train=np.reshape(y_train,(-1,1))


# just for using less data
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X_train,y_train,stratify=y_train,test_size=5/6,random_state=42)

col_mean = X_train.mean(axis=0)
X_train = X_train - col_mean
col_std = X_train.std(axis=0)
for i in range(X_train.shape[1]):
    if(col_std[i] != 0):
        X_train[:,i] = X_train[:,i]/(col_std[i])


eps=1e-6

def add_cov(cov):
    tmp=cov
    for i in range(len(cov)):
        tmp[i][i]+=1e-8
    return tmp

def pdf(x,mu,incov):
    n=len(x)
    x=np.reshape(x,(-1,1))
    mu=np.reshape(mu,(-1,1))
    t2=np.dot(np.dot((x-mu).T,incov),(x-mu))
    t2=-0.5*np.reshape(t2,(t2.shape[0],))
    return exp(t2)

def pdfvec(data,mu,incov):
    pdfs=[]
    for i in range(len(data)):
        pdfs.append(pdf(data[i],mu,incov))
    pdfs=np.array(pdfs)
    pdfs=np.reshape(pdfs,(-1,1))
    return pdfs


# intialization
from sklearn.datasets import make_spd_matrix
k=10
n,d=X_train.shape[0],X_train.shape[1]
weights=np.ones((k,1))/k
covs=[]
for i in range(k+1):
    covs.append(make_spd_matrix(d))
covs.pop(0)
covs=np.array(covs)
mus=np.random.rand(k,d,1)
b=np.zeros((n,k))
pdfs=np.zeros((n,k))


eps=1e-8
for step in range(50):
    for j in range(k):
        pdfs[:,j]=pdfvec(X_train,mus[j],inv(add_cov(covs[j]))).T
    for i in range(n):
        denom=np.dot(pdfs[i],weights)+eps
        for j in range(k):
            b[i][j]=pdfs[i][j]*weights[j]/denom
    for j in range(k):
        nk=np.sum(b[:,j])
        col=np.reshape(b[:,j],(n,1))
        mus[j]=np.reshape(np.sum(np.multiply(col,X_train),axis=0)/nk,(-1,1))
        new_cov=np.zeros((d,d))
        for i in range(n):
            x=np.reshape(X_train[i],(-1,d))
            new_cov=new_cov + b[i][j]*np.dot((x-mus[j]),(x-mus[j]).T)
        new_cov=new_cov/nk
        covs[j]=new_cov
        weights[j]=nk/n
    print("step "+str(step+1)+" completed")


# image generation
for i in range(k):
    sample=np.random.multivariate_normal(np.reshape(mus[i],(784,)),covs[i],1)
    x=pca.inverse_transform(sample)
    x=np.reshape(x,(28,28))
    x[x<0.1]=0
    plt.imshow(x, cmap='gray')
    plt.show()
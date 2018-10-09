"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import csv
import numpy
from sklearn import preprocessing  
import urllib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_data(path,type_,size,dataset):
    
   if dataset=='authentication':
       
     if size==50:
       reg=np.load('./dataset/authentic/'+str(size)+'/train/reg_data'+str(size)+'.npy')
       mal=np.load('./dataset/authentic/'+str(size)+'/train/mal_data'+str(size)+'.npy')
     if size==300:
       reg=np.load('./dataset/authentic/'+str(size)+'/train/reg_data.npy')
       mal=np.load('./dataset/authentic/'+str(size)+'/train/mal_data.npy')
       
     data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
     for i in range(reg.shape[0]):
       data[i,:,:,0]=reg[i]
       data[i,:,:,1]=mal[i]
       if size==50:
          return data[:37]
          #return data[37:]   ----cross validation
       if size==300:
          return data[:249]
          #return np.concatenate((data[:136],data[249:]),axis=0)   
          #return data[136:]           --cross validation
          
          
   if dataset=='scale-free':
     reg=np.load('./dataset/scale free/'+str(size)+'/scale_reg'+str(size)+'.npy')
     mal=np.load('./dataset/scale free/'+str(size)+'/scale_mal'+str(size)+'.npy')
     data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
     for i in range(reg.shape[0]):
       data[i,:,:,0]=reg[i]
       data[i,:,:,1]=mal[i]
     return data[:2500] 
 
   if dataset=='poisson-random':
     reg=np.load('./dataset/poisson random/'+str(size)+'/poisson_reg'+str(size)+'.npy')
     mal=np.load('./dataset/poisson random/'+str(size)+'/poisson_mal'+str(size)+'.npy')
     data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
     for i in range(reg.shape[0]):
       data[i,:,:,0]=reg[i]
       data[i,:,:,1]=mal[i]
     return data[:2500]    
   
def load_data_test_auth(path,filename,size):
    
    if size==50:
      reg=np.load('./dataset/authentic/'+str(size)+'/test/'+filename+'_50.npy')[:2400]
    if size==300:
      reg=np.load('./dataset/authentic/'+str(size)+'/test/'+filename+'.npy')[:2400]
      
    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
      data[i,:,:,0]=reg[i]
      data[i,:,:,1]=reg[i]
    return data

def load_data_test(size,dataset):
    
    if dataset=='scale-free':
      reg=np.load('./dataset/scale free/'+str(size)+'/scale_reg'+str(size)+'.npy')[2500:]   
    if dataset=='poisson-random':
      reg=np.load('./dataset/poisson random/'+str(size)+'/poisson_reg'+str(size)+'.npy')[2500:]
    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
      data[i,:,:,0]=reg[i]
      data[i,:,:,1]=reg[i]
    return data




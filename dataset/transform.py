# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:13:10 2018

@author: gxjco
"""
import numpy as np

def di_2_undi(filename):
   s1=np.load(filename)
   n_node=s1.shape[1]
   s2=np.zeros((len(s1),2*n_node,2*n_node))
   for i in range(len(s2)):
       for m in range(n_node):
           for n in range(n_node):
               if n<m:s2[i][2*n_node-m-1][2*n_node-n-1]=s1[i][m][n]
               if n>=m: s2[i][m][n]=s1[i][m][n]
   np.save(filename.split('.')[0]+'_undirected.npy',s2)
   return s2

#di_2_undi('mal_data.npy')
#di_2_undi('reg_data.npy')

def undi_2_di(filename):
    adj=np.load(filename)
    n_node=int(adj.shape[1]/2)
    adj1=np.zeros((len(adj),n_node,n_node))
    for m in range(len(adj)):
     for i in range(n_node):
        for j in range(n_node):
            if j>=i: adj1[m][i][j]=adj[m][i][j]
            if j<i: adj1[m][i][j]=adj[m][2*n_node-i-1][2*n_node-j-1]
    np.save(filename.split('.')[0]+'_directed.npy',adj1)
    


'''
with open('C:/Users/gxjco/Desktop/iclr2019/dataset/authentic/300/name_list.txt') as f:
    name=f.readlines()
    name_list=[]
    for i in range(len(name)):
        if name[i].split('\n')[0] not in name_list:
           name_list.append(name[i].split('\n')[0])
for i in range(73,len(name_list)):
  di_2_undi('C:/Users/gxjco/Desktop/iclr2019/dataset/authentic/150/'+name_list[i]+'.npy')
  di_2_undi('C:/Users/gxjco/Desktop/iclr2019/dataset/authentic/150/'+name_list[i]+'_mal.npy')'''
  
#di_2_undi('C:/Users/gxjco/Desktop/iclr2019/dataset/scale free/150/scale_mal150.npy')
di_2_undi('C:/Users/gxjco/Desktop/iclr2019/dataset/scale free/150/scale_reg150.npy')  
 
#undi_2_di('C:/Users/gxjco/Desktop/iclr2019/dataset/scale free/gen150.npy')

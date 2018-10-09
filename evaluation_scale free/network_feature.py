# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:52:58 2018

@author: gxjco
"""

import networkx as nx
import numpy as np
import os
with open('C:/Users/gxjco/Desktop/redteam.txt/redteam.txt') as f:
    red_team=f.readlines()
red=[]
red_user={}
for i in red_team:
    act={}
    m=i.split(',')
    act['time']=m[0]
    act['user']=m[1].split('@')[0]
    act['s_c']=m[2]
    act['d_c']=m[3].split('\n')[0]
    red.append(act)
    if act['user'] not in red_user:
        red_user[act['user']]=[]
for act_ in red:
    red_user[act_['user']].append(act_)
def plus(a,b):
    for i in b:
        if i not in a:
            a.append(i)
    return a
def feature(node,edge):
  G =nx.MultiGraph()
  G=G.to_directed()
  G.add_nodes_from(node)
  G.add_edges_from(edge)
  f=np.zeros(10)
  f[0]=len(G.nodes)
  f[1]=len(G.edges)
  f[2]=nx.density(G)
  f[3]=nx.degree_pearson_correlation_coefficient(G)
  f[4]=nx.algorithms.reciprocity(G)
  f[5]=0#nx.transitivity(G)
  f[6]=nx.is_weakly_connected(G)
  f[7]=nx.number_weakly_connected_components(G)
  f[8]=nx.is_strongly_connected(G)
  f[9]=nx.number_strongly_connected_components(G)
  return f
b=0
path='C:/Users/gxjco/Desktop/redteam.txt/user_new/'
files=os.listdir(path)
for file_ in files:
    u=[]
    user=file_.split('.')[0]
    with open(path+file_) as f:
       for line in f:
          m=line.split(',')
          event={}        
          event['time']=m[0]
          event['user']=m[1].split('@')[0]
          event['s_c']=m[3]
          event['d_c']=m[4]
          u.append(event) 
    l=red_user[user]
    u=plus(u,l)   
    act={}
    for i in range(2600):
        act[i]={}
        act[i]['node']=[]
        act[i]['edge']=[]                
    for line in u:
        if int(line['time'])<4680000:
             act[int(int(line['time'])/1800)]['node'].append(line['s_c'])
             act[int(int(line['time'])/1800)]['node'].append(line['d_c'])
             act[int(int(line['time'])/1800)]['edge'].append((line['s_c'],line['d_c']))
    sample=np.zeros((2600,10))
    for i in range(2600):
        try:
          sample[i,:]=feature(act[i]['node'],act[i]['edge'])
        except:b+=1
        
    time_red={}
    for i in red_user[user]:
       time_red[i['time']]=[]
    label=[]
    for t in range(2600):
      o=0
      for time in time_red:
        if int(time) in range(1800*t,1800*t+1800):
            o=1
            break
      label.append(o)
    #seperate the negative and positive
    p_data=[]
    n_data=[]
    for i in range(2600):
      if label[i]==1:
         p_data.append(sample[i])
      else:
         n_data.append(sample[i])
    p_sample=np.zeros((len(p_data),10))
    n_sample=np.zeros((len(n_data),10))
    for i in range(len(p_data)):
        p_sample[i]=p_data[i]
    for i in range(len(n_data)):
        n_sample[i]=n_data[i]
    np.save('C:/Users/gxjco/Desktop/redteam.txt/input_new/feature_new_negative_sample__'+user+'.npy',n_sample)
    np.save('C:/Users/gxjco/Desktop/redteam.txt/input_new/feature_new_positive_sample__'+user+'.npy',p_sample)
           

  
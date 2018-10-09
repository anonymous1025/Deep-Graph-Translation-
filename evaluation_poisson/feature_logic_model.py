# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:57:38 2018

@author: gxjco
"""
import numpy as np
from imblearn.over_sampling import SMOTE
import xlwt
import xlrd  
import os  
from xlutils.copy import copy  
from xlwt import Style 
from sklearn.linear_model import LogisticRegression 
import math 
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
def read_data(user,if_smote):
  n_data_=np.load('C:/Users/gxjco/Desktop/redteam.txt/input_feature/feature_new_negative_sample__'+user+'.npy')   
  p_data_=np.load('C:/Users/gxjco/Desktop/redteam.txt/input_feature/feature_new_positive_sample__'+user+'.npy')
  for i in range(len(n_data_)):
      if math.isnan(n_data_[i,3]): n_data_[i,3]=0
  for i in range(len(p_data_)):
      if math.isnan(p_data_[i,3]): p_data_[i,3]=0 
  if len(p_data_)>1:
    data=np.concatenate((n_data_,p_data_),axis=0)
    #data=np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
    label=np.concatenate((np.zeros(len(n_data_),dtype=np.int32),np.ones(len(p_data_),dtype=np.int32)),axis=0)
    sm=SMOTE(k_neighbors=len(p_data_)-1)
    data_res, label_res = sm.fit_sample(data, label)
    n=data_res[0:int(len(data_res)/2)]
    p=data_res[int(len(data_res)/2):]
    n_data=n
    p_data=p
  else:
    n_data=n_data_#np.reshape(n_data_,[n_data_.shape[0],n_data_.shape[1]*n_data_.shape[2]])
    p_data_=p_data_#np.reshape(p_data_,[p_data_.shape[0],p_data_.shape[1]*p_data_.shape[2]])
    p_data=n_data
    for i in range(len(p_data)):
        p_data[i,:]=p_data_[0]      
  size_n=int(len(n_data)/2)
  size_p=int(len(p_data)/2)
  n_label=np.zeros(len(n_data))
  p_label=np.ones(len(p_data))      
  train_d=np.concatenate((n_data[0:size_n],p_data[0:size_p]), axis=0)
  test_d=np.concatenate((n_data[size_n:],p_data[size_p:]), axis=0)
  train_l=np.concatenate((n_label[0:size_n],p_label[0:size_p]), axis=0)
  test_l=np.concatenate((n_label[size_n:],p_label[size_p:]), axis=0)
  return train_d,test_d,train_l,test_l

def result(predictions, labels,score):
     r11=0
     r00=0
     r10=0
     r01=0
     for i in range(len(predictions)):
         if predictions[i]==1 and labels[i]==1: r11+=1
         if predictions[i]==0 and labels[i]==0: r00+=1
         if predictions[i]==1 and labels[i]==0: r10+=1
         if predictions[i]==0 and labels[i]==1: r01+=1
     fpr, tpr, thresholds = metrics.roc_curve(labels,score,pos_label=1)
     AUC=metrics.auc(fpr, tpr)
     precision=precision_score(labels,predictions)
     recall=recall_score(labels,predictions)
     return r11,r00,r01,r10,AUC,precision,recall
def writeExcel(row, col, value, file_name,styl=Style.default_style):  
    rb = xlrd.open_workbook(file_name)  
    wb = copy(rb)  
    ws = wb.get_sheet(0)  
    ws.write(row, col, value, styl)  
    wb.save(file_name)  
def main(user):
  train_data,test_data,train_label,test_label=read_data(user,True)
  classifier = LogisticRegression()  
  classifier.fit(train_data, train_label)
  prediction = classifier.predict(test_data)
  score=classifier.predict_proba(test_data)
  result_={}
  result_['TP'],result_['TN'],result_['FN'],result_['FP'],result_['AUC'],result_['precision'],result_['recall']=result(prediction, test_label,score[:,1])
  return result_,test_label,score[:,1]

with open('redteam.txt') as f:
    red_team=f.readlines()
red=[]
red_user={}
for i in red_team:
    act={}
    m=i.split(',')
    act['time']=str(m[0])
    act['user']=m[1].split('@')[0]
    act['s_c']=m[2]
    act['d_c']=m[3].split('\n')[0]
    red.append(act)
    if act['user'] not in red_user:
        red_user[act['user']]=[]        
num=0
skip1=[]
lab=[]
sc=[]
for user in red_user:
   try:
    acc,l,s=main(user)
    #writeExcel(int(num)+1,0, user, 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,1, acc['TN'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,2, acc['TP'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,3, acc['FN'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,4, acc['FP'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,5, acc['AUC'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,6, acc['precision'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    #writeExcel(int(num)+1,7, acc['recall'], 'C:/Users/gxjco/Desktop/redteam.txt/feature_result_logic.xls')
    print(num)
    num+=1
    lab.append(l)
    sc.append(s)
   except: skip1.append(user)
   
label_total=lab[0]
score_total=sc[0]
for i in range(97):
    label_total=np.concatenate((label_total,lab[i+1]),axis=0)
    score_total=np.concatenate((score_total,sc[i+1]),axis=0)
#draw ROC tota
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr,tpr,threshold = roc_curve(label_total, score_total)   
roc_auc = auc(fpr,tpr)
plt.figure()  
lw = 2  
plt.figure(figsize=(10,10))  
plt.plot(fpr, tpr, color='darkorange',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic on 10_feature model')  
plt.legend(loc="lower right")  
plt.show()  
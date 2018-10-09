# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:43:11 2018

@author: gxjco
"""
import xlwt
import xlrd 
from xlutils.copy import copy  
from xlwt import Style 
from main import*
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#find user 
with open('./dataset/authentic/50/name_list50.txt') as f:  #change the 50 to 300 when size changes
   user_name=f.readlines()
name_list=[]
for i in user_name:
    if i.split('\n')[0] not in name_list: name_list.append(i.split('\n')[0])

def writeExcel(row, col, value, file_name,styl=Style.default_style):  
    rb = xlrd.open_workbook(file_name)  
    wb = copy(rb)  
    ws = wb.get_sheet(0)  
    ws.write(row, col, value, styl)  
    wb.save(file_name) 
num=0
lab_fake=[]
sc_fake=[]
lab_real=[]
sc_real=[]
skip=[]
for user in name_list:   
   try:
    TP_fake,TN_fake,FP_fake,FN_fake,precision_fake,recall_fake,l_fake,s_fake, TP_real,TN_real,FP_real,FN_real,precision_real,recall_real,l_real,s_real=test_(user)
    writeExcel(int(num)+1,0, user, './result_GAN_50.xls')
    writeExcel(int(num)+1,1, TN_fake, './result_GAN_50.xls')
    writeExcel(int(num)+1,2, TP_fake, './result_GAN_50.xls')
    writeExcel(int(num)+1,3, FN_fake, './result_GAN_50.xls')
    writeExcel(int(num)+1,4, FP_fake, './result_GAN_50.xls')
    writeExcel(int(num)+1,5, precision_fake, './result_GAN_50.xls')
    writeExcel(int(num)+1,6, recall_fake, './result_GAN_50.xls')
    
    writeExcel(int(num)+1,8, user, './result_GAN_50.xls')
    writeExcel(int(num)+1,9, TN_real, './result_GAN_50.xls')
    writeExcel(int(num)+1,10, TP_real, './result_GAN_50.xls')
    writeExcel(int(num)+1,11, FN_real, './result_GAN_50.xls')
    writeExcel(int(num)+1,12, FP_real, './result_GAN_50.xls')
    writeExcel(int(num)+1,13, precision_real, './result_GAN_50.xls')
    writeExcel(int(num)+1,14, recall_real, './result_GAN_50.xls')
    print(num)
    num+=1
    lab_fake.append(l_fake)
    sc_fake.append(s_fake)
    lab_real.append(l_real)
    sc_real.append(s_real)
   except: skip.append(user)
label_total_fake=lab_fake[0]
score_total_fake=sc_fake[0]
label_total_real=lab_real[0]
score_total_real=sc_real[0]
for i in range(46):  
    label_total_fake=np.concatenate((label_total_fake,lab_fake[i+1]),axis=0)
    score_total_fake=np.concatenate((score_total_fake,sc_fake[i+1]),axis=0)
    label_total_real=np.concatenate((label_total_real,lab_real[i+1]),axis=0)
    score_total_real=np.concatenate((score_total_real,sc_real[i+1]),axis=0)

#draw ROC tota
def draw_roc(label_total_real, score_total_real,label_total_fake, score_total_fake):

 fpr1,tpr1,threshold1 = roc_curve(label_total_fake, score_total_fake)   
 roc_auc1 = auc(fpr1,tpr1)
 fpr2,tpr2,threshold2 = roc_curve(label_total_real, score_total_real)   
 roc_auc2 = auc(fpr2,tpr2)
 plt.figure()  
 lw = 2  
 plt.figure(figsize=(10,10))  
 plt.plot(fpr1, tpr1, color='darkorange',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc1) 
 plt.plot(fpr2, tpr2, color='green',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2) 
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
 plt.xlim([0.0, 1.0])  
 plt.ylim([0.0, 1.05])  
 plt.xlabel('False Positive Rate')  
 plt.ylabel('True Positive Rate')  
 plt.title('ROC curve on KCNN model')  
 plt.legend(loc="lower right")  
 plt.show()    
draw_roc(label_total_real, score_total_real,label_total_fake, score_total_fake)
roc=[]
'''
for i in range(49):
    j=name_list.index(name_list50[i])
    from sklearn.metrics import roc_curve, auc
    fpr1,tpr1,threshold1 = roc_curve(label_total_fake[j*2400:j*2400+2400], score_total_fake[j*2400:j*2400+2400])   
    roc.append(auc(fpr1,tpr1))'''
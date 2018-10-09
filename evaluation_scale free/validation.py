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



def writeExcel(row, col, value, file_name,styl=Style.default_style):  
    rb = xlrd.open_workbook(file_name)  
    wb = copy(rb)  
    ws = wb.get_sheet(0)  
    ws.write(row, col, value, styl)  
    wb.save(file_name) 

TP_fake,TN_fake,FP_fake,FN_fake,precision_fake,recall_fake,l_fake,s_fake, TP_real,TN_real,FP_real,FN_real,precision_real,recall_real,l_real,s_real=test_()


#draw ROC tota
def draw_roc(label_total_real, score_total_real,label_total_fake, score_total_fake):
 from sklearn.metrics import roc_curve, auc
 import matplotlib.pyplot as plt
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
draw_roc(l_real, s_real,l_fake, s_fake)
from sklearn.metrics import precision_score,recall_score
s=s_fake
l=[]
for i in range(len(s)):
    if s[i]>1e-20: l.append(1)
    else: l.append(0)
print(precision_score(l_real, l))
print(recall_score(l_real, l))
import torch 
import torch.nn as nn
#import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np
from utils import compute_nystrom,create_train_test_loaders,imbalance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from model import CNN
from graph_kernels import sp_kernel, wl_kernel
import xlwt
import xlrd 
from xlutils.copy import copy  
from xlwt import Style 
def writeExcel(row, col, value, file_name,styl=Style.default_style):  
    rb = xlrd.open_workbook(file_name)  
    wb = copy(rb)  
    ws = wb.get_sheet(0)  
    ws.write(row, col, value, styl)  
    wb.save(file_name) 

community_detection = "louvain"

# Hyper Parameters
dim = 100
batch_size = 64
num_epochs = 150
num_filters = 50
hidden_size = 50
learning_rate = 0.005
use_node_labels = False
# Choose kernels
kernels=[wl_kernel]
num_kernels = len(kernels)

def test_(user):
  print("Computing feature maps...")
  Q, subgraphs, labels,shapes = compute_nystrom(user, use_node_labels, dim, community_detection, kernels)
  M=np.zeros((shapes[0],shapes[1],len(kernels)))
  for idx,k in enumerate(kernels):
    M[:,:,idx]=Q[idx]
  Q=M
  # Binarize labels
  le = LabelEncoder()
  y = le.fit_transform(labels)
  # Build vocabulary
  max_document_length = max([len(x.split(" ")) for x in subgraphs])
  x = np.zeros((len(subgraphs), max_document_length), dtype=np.int32)
  for i in range(len(subgraphs)):
	  communities = subgraphs[i].split()
	  for j in range(len(communities)):
		  x[i,j] = int(communities[j])
  reg=x[0:2400]
  gen=x[2400:4800]
  mal=x[4800:]        
  reg_label=y[:2400]
  gen_label=y[2400:4800]
  mal_label=y[4800:]
  
  X,Y=imbalance(np.concatenate((reg,mal),axis=0),np.concatenate((reg_label,mal_label),axis=0))
  train_reg=X[0:1200]
  test_reg=X[1200:2400]
  train_reg_y=Y[0:1200]
  test_reg_y=Y[1200:2400]
  train_mal=X[2400:3600]
  test_mal=X[3600:4800]
  train_mal_y=Y[2400:3600]
  test_mal_y=Y[3600:4800]
  train_gen=gen[0:1200]
  train_gen_y=gen_label[0:1200]
  
  train_fake=np.concatenate((train_reg,train_gen),axis=0)
  y_train_fake=np.concatenate((train_reg_y,train_gen_y),axis=0)
  train_real=np.concatenate((train_reg,train_mal),axis=0)
  y_train_real=np.concatenate((train_reg_y,train_mal_y),axis=0)
  test=np.concatenate((test_reg,test_mal),axis=0)
  y_test=np.concatenate((test_reg_y,test_mal_y),axis=0)

      
  def train_test(Q, x_train, x_test, y_train, y_test, batch_size):   
    train_loader, test_loader = create_train_test_loaders(Q, x_train, x_test, y_train, y_test, batch_size)		   
    cnn = CNN(input_size=num_filters, hidden_size=hidden_size, num_classes=np.unique(y).size, dim=dim, num_kernels=num_kernels, max_document_length=max_document_length)
    if torch.cuda.is_available():
         cnn.cuda()
    if torch.cuda.is_available():
         criterion = nn.CrossEntropyLoss().cuda()
    else:
         criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
       for i, (graphs, labels) in enumerate(train_loader):
           graphs = Variable(graphs)
           labels = Variable(labels)
           optimizer.zero_grad()
           outputs = cnn(graphs)
           if torch.cuda.is_available():
              loss = criterion(outputs, labels.cuda())
           else:
              loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
		    	# Test the Model
                
    cnn.eval() 
    correct = 0
    total = 0
    TP=0
    TN=0
    FP=0
    FN=0
    predict=[]
    label=[]
    output=[]
    for graphs, labels in test_loader:
       graphs = Variable(graphs)
       outputs = cnn(graphs)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels.cuda()).sum()
       TP += (predicted+labels.cuda()==2).sum()
       FP+=(predicted*5+labels.cuda()*1==5).sum()
       FN+=(predicted*1+labels.cuda()*5==5).sum()
       TN+=(predicted+labels.cuda()==0).sum()
       predict.append(predicted)
       label.append(labels)
       output.append(outputs.data)
    if TP+FP==0: precision=0
    else: precision=TP/(TP+FP)
    if TP+FN==0: recall=0
    else: recall=TP/(TP+FN)
    l=np.zeros((len(label)))
    for i in range(len(label)):
        l[i]=int(label[i])
    s=np.zeros((len(output)))
    for i in range(len(output)):
        s[i]=output[i][0][1]
    return TP,TN,FP,FN,precision,recall,l,s
  TP_fake,TN_fake,FP_fake,FN_fake,precision_fake,recall_fake,l_fake,s_fake=train_test(Q, train_fake, test, y_train_fake, y_test, batch_size)
  TP_real,TN_real,FP_real,FN_real,precision_real,recall_real,l_real,s_real=train_test(Q, train_real, test, y_train_real, y_test, batch_size)
  return TP_fake,TN_fake,FP_fake,FN_fake,precision_fake,recall_fake,l_fake,s_fake,TP_real,TN_real,FP_real,FN_real,precision_real,recall_real,l_real,s_real
'''
num=0
lab=[]
sc=[]
skip=[]
for user in red_user:   
   try:
    TP,TN,FP,FN,precision,recall,l,s=test_(user,red_user)
    writeExcel(int(num)+1,0, user, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,1, TN, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,2, TP, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,3, FN, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,4, FP, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,5, precision, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    writeExcel(int(num)+1,6, recall, 'C:/Users/gxjco/Desktop/New folder (2)/result_CGN.xls')
    print(num)
    num+=1
    lab.append(l)
    sc.append(s)
   except: skip.append(user)
  
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
plt.title('ROC curve on KCNN model')  
plt.legend(loc="lower right")  
plt.show() '''
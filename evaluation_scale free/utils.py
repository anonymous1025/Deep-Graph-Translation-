import random
import networkx as nx
import igraph as ig
import numpy as np
from nystrom import Nystrom
import torch
import torch.utils.data as utils
from imblearn.over_sampling import SMOTE
'''
def load(user,red_user):
  def plus(a,b):
    for i in b:
        if i not in a:
            a.append(i)
    return a
  path='C:/Users/gxjco/Desktop/New folder (2)/red_user_data_new/user/'
  with open(path+user+'.txt') as f:
    u=[]
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
    Gs = []
    for i in range(2600):
        G =nx.MultiGraph()
        G=G.to_directed()
        G.add_nodes_from(act[i]['node'])
        G.add_edges_from(act[i]['edge'])
        Gs.append(G)
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
    labels  = np.array(label, dtype = np.float)
    return Gs, labels'''

def load():
  def graph(A):
    G =nx.MultiGraph()
    G=G.to_directed()
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
             if A[i][j]>0: 
                 G.add_nodes_from([str(i),str(j)])
                 for m in range(A[i][j]):
                   G.add_edges_from([(str(i),str(j))])
                
    return G
  reg=np.load('./dataset/scale free/10/'+'scale_reg10.npy').astype(int)[2500:]
  gen=np.load('./scale_gen10.npy').astype(int)[:,:,:,0]
  mal=np.load('./dataset/scale free/10/'+'scale_mal10.npy').astype(int)[2500:]
  Gs_reg=[]
  Gs_gen=[]
  Gs_mal=[]
  for i in range(len(reg)):
      Gs_reg.append(graph(reg[i]))
      Gs_gen.append(graph(gen[i]))
  for i in range(len(mal)):
      Gs_mal.append(graph(mal[i]))
  label_reg=np.zeros(len(reg))
  label_gen=np.ones(len(gen))
  label_mal=np.ones(len(mal))
  return Gs_reg,label_reg,Gs_gen,label_gen,Gs_mal,label_mal
   
	

def networkx_to_igraph(G):
	mapping = dict(zip(G.nodes(),range(G.number_of_nodes())))
	reverse_mapping = dict(zip(range(G.number_of_nodes()),G.nodes()))
	G = nx.relabel_nodes(G,mapping)
	G_ig = ig.Graph(len(G), list(zip(*list(zip(*nx.to_edgelist(G)))[:2])))
	return G_ig, reverse_mapping


def community_detection(G_networkx, community_detection_method):
	G,reverse_mapping = networkx_to_igraph(G_networkx)	
	if community_detection_method == "eigenvector":
		c = G.community_leading_eigenvector()
	elif community_detection_method == "infomap":
		c = G.community_infomap()
	elif community_detection_method == "fastgreedy":
		c = G.community_fastgreedy().as_clustering()
	elif community_detection_method == "label_propagation":
		c = G.community_label_propagation()
	elif community_detection_method == "louvain":
		c = G.community_multilevel()
	elif community_detection_method == "spinglass":
		c = G.community_spinglass()
	elif community_detection_method == "walktrap":
		c = G.community_walktrap().as_clustering()
	else:
		c = []
	
	communities = []	
	for i in range(len(c)):
		community = []
		for j in range(len(c[i])):
			community.append(reverse_mapping[G.vs[c[i][j]].index])
		
		communities.append(community)

	return communities

def compute_communities(graphs, use_node_labels, community_detection_method):
	communities = []
	subgraphs = []
	counter = 0
	coms = []
	for G in graphs:
		c = community_detection(G, community_detection_method)
		coms.append(len(c))
		subgraph = []
		for i in range(len(c)):
			communities.append(G.subgraph(c[i]))
			subgraph.append(counter)
			counter += 1

		subgraphs.append(' '.join(str(s) for s in subgraph))

	return communities, subgraphs
	
   
def compute_nystrom(use_node_labels, embedding_dim, community_detection_method, kernels):
   graphs_reg, labels_reg,graphs_gen, labels_gen,graphs_mal, labels_mal = load()
   graphs=graphs_reg+graphs_gen+graphs_mal
   labels=np.concatenate((labels_reg,labels_gen,labels_mal),axis=0)
   communities, subgraphs = compute_communities(graphs, use_node_labels, community_detection_method)

   print("Number of communities: ", len(communities))
   lens = []
   for community in communities:
       lens.append(community.number_of_nodes())

   print("Average size: %.2f" % np.mean(lens))
   Q=[]
   for idx, k in enumerate(kernels):
       model = Nystrom(k, n_components=embedding_dim)
       model.fit(communities)
       Q_t = model.transform(communities)
       Q_t = np.vstack([np.zeros(embedding_dim), Q_t])
       Q.append(Q_t)

   return Q, subgraphs, labels, Q_t.shape


def create_train_test_loaders(Q, x_train, x_test, y_train, y_test, batch_size):
	num_kernels = Q.shape[2]
	max_document_length = x_train.shape[1]
	dim = Q.shape[1]
	
	my_x = []
	for i in range(x_train.shape[0]):
		temp = np.zeros((1, num_kernels, max_document_length, dim))
		for j in range(num_kernels):
			for k in range(x_train.shape[1]):
				temp[0,j,k,:] = Q[x_train[i,k],:,j].squeeze()
		my_x.append(temp)

	if torch.cuda.is_available():
		tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x]) # transform to torch tensors
		tensor_y = torch.cuda.LongTensor(y_train.tolist())
	else:
		tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
		tensor_y = torch.from_numpy(np.asarray(y_train,dtype=np.int64))

	train_dataset = utils.TensorDataset(tensor_x, tensor_y)
	train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	my_x = []
	for i in range(x_test.shape[0]):
		temp = np.zeros((1, num_kernels, max_document_length, dim))
		for j in range(num_kernels):
			for k in range(x_test.shape[1]):
				temp[0,j,k,:] = Q[x_test[i,k],:,j].squeeze()
		my_x.append(temp)

	if torch.cuda.is_available():
		tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x]) # transform to torch tensors
		tensor_y = torch.cuda.LongTensor(y_test.tolist())
	else:
		tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
		tensor_y = torch.from_numpy(np.asarray(y_test,dtype=np.int64))

	test_dataset = utils.TensorDataset(tensor_x, tensor_y)
	test_loader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

	return train_loader, test_loader

def imbalance(x,y):
  p_num=sum(y)
  if p_num>1:
    sm=SMOTE(k_neighbors=p_num-1)
    data_res, label_res = sm.fit_sample(x, y)
  else:
    for i in range(2600):
      if y[i]==1: 
          idx=i
          break
    new_p=np.zeros((2598,x.shape[1]))
    new_p_l=np.ones((2598))
    for i in range(2598):
        new_p[i,:]=x[idx]
    data_res=np.concatenate((x,new_p),axis=0)
    label_res=np.concatenate((y,new_p_l),axis=0)
  return data_res.astype(dtype=np.int32),label_res.astype(dtype=np.int32)
    
    
      
      
      
      
      
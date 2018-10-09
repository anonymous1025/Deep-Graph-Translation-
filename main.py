# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:40:35 2018

@author: gxjco
"""
import argparse
import os
import scipy.misc
import numpy as np
from model import graph2graph
import tensorflow as tf
import datetime

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=450, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='# graphs in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# graphs used to train')
parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output channels')
parser.add_argument('--niter', dest='niter', type=int, default=100, help='# of iter at starting learning rate') 
parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.0001  , help='initial learning rate for adam')
parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.00005  , help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='m   omentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the graphs for data argumentation')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes graphsin order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial graph list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_auth_50', help='models are saved here,need to be distinguishable for different dataset')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./validation_data_auth_50/', help='test sample are saved here, need to be distinguishable for different dataset')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--train_dir', dest='train_dir', default='./', help='train sample are saved here')
parser.add_argument('--graph_size', dest='graph_size', default=[50,50], help='size of graph')
parser.add_argument('--output_size', dest='output_size', default=[50,50], help='size of graph')
parser.add_argument('--dataset', dest='dataset', default='authentication', help='chose from authentication, scale-free and poisson-random')
args = parser.parse_args()


def main_auth(filename):
    #start = datetime.datetime.now()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    tf.reset_default_graph() 
    with tf.Session() as sess:
        model = graph2graph(sess, batch_size=args.batch_size,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir,test_dir=args.test_dir,train_dir=args.train_dir,graph_size=args.graph_size,output_size=args.output_size)
        #model.train(args)
        model.test_auth(args,filename)
    #end = datetime.datetime.now()
    #print (end-start)


def main():
      #start = datetime.datetime.now()
      if not os.path.exists(args.checkpoint_dir):
         os.makedirs(args.checkpoint_dir)
      if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
      if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
      tf.reset_default_graph() 
      with tf.Session() as sess:
         model = graph2graph(sess, batch_size=args.batch_size,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir,test_dir=args.test_dir,train_dir=args.train_dir,graph_size=args.graph_size,output_size=args.output_size)
         #model.train(args)
         model.test(args)
      #end = datetime.datetime.now()
      #print (end-start)        

if __name__ == '__main__':
    
   if args.dataset=='authentication':
       if args.graph_size[0]==50:
          with open('./dataset/authentic/50/name_list50_1.txt') as f:
            user_name=f.readlines()
            name_list=[]
            for i in user_name:
              if i.split('\n')[0] not in name_list: name_list.append(i.split('\n')[0])
              
       if args.graph_size[0]==300:
          with open('./dataset/authentic/300/name_list1.txt') as f:
            user_name=f.readlines()
            name_list=[]
            for i in user_name:
              if i.split('\n')[0] not in name_list: name_list.append(i.split('\n')[0])   
              
       for i in name_list:
            main_auth(i)
            
   if args.dataset=='scale-free' or 'poisson-random': 
        main()
      #tf.app.run()
    
#from memory_profiler import memory_usage
 #   m=memory_usage(main(), interval=20, timeout=240)
  #  print(max(m))'''

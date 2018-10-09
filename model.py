# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:46:42 2018

@author: gxjco
"""


from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import csv
from ops_ import *
from utils_ import *
from sklearn.metrics import mean_squared_error
from math import sqrt

class graph2graph(object):
    def __init__(self, sess, test_dir,train_dir,graph_size,output_size,dataset,
                 batch_size=50, sample_size=1, 
                 gf_dim=10, df_dim=10, L1_lambda=100,
                 input_c_dim=1, output_c_dim=1,
                 checkpoint_dir=None, sample_dir=None,g_train_num=6,d_train_num=6):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the graphs. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input graph channel. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output graph channel. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.g_train_num=g_train_num
        self.d_train_num=d_train_num
        self.test_dir=test_dir
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.dataset=dataset
        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

     
        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.graph_size[0], self.graph_size[1],
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_graphs')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A)
        
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)   #define the input from graph
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.graph("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_AB - self.fake_AB))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self,sample_dir):
        sample_data=load_data(sample_dir)
        sample = np.random.choice(sample_data, self.batch_size)
        sample_graphs = np.array(sample).astype(np.float32)
        return sample_graphs

    def sample_model(self, sample_dir, epoch, idx):
        sample_graphs = self.load_random_samples(sample_dir)
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_graphs}
        )
       # save_graphs(samples, [self.batch_size, 1],
        #            './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        #d_optim = tf.train.GradientDescentOptimizer(args.lr) \
         #                 .minimize(self.d_loss, var_list=self.d_vars)
        d_optim = tf.train.AdamOptimizer(args.lr_d, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)               
        #g_optim = tf.train.GradientDescentOptimizer(args.lr) \
         #                 .minimize(self.g_loss, var_list=self.g_vars)                
        g_optim = tf.train.AdamOptimizer(args.lr_g, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        data = load_data(args.train_dir,'train',self.graph_size[0],self.dataset)
        errD_fake = 0
        errD_real = 0
        errG = 0
        best=5
        for epoch in xrange(args.epoch):           
            #np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_graphs = np.array(batch).astype(np.float32)
                if errD_fake+errD_real>0.5:
                  for i in range(self.d_train_num):                 
                     # Update D network
                     _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_graphs })
                     self.writer.add_summary(summary_str, counter)
                
                for i in range(self.g_train_num):
                    # Update G network
                     _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_graphs })
                     self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)


                errD_fake = self.d_loss_fake.eval({self.real_data: batch_graphs})
                errD_real = self.d_loss_real.eval({self.real_data: batch_graphs})
                errG = self.g_loss.eval({self.real_data: batch_graphs})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

               # if np.mod(counter, 100) == 1:
                #    self.sample_model(args.sample_dir, epoch, idx)

                if errG<best and errD_fake+errD_real<0.7:
                    self.save(args.checkpoint_dir, counter)
                    best=errG

    def discriminator(self, graph, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # graph is 300 x 300 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = lrelu(e2e(graph, self.df_dim,k_h=self.graph_size[0], name='d_h0_conv'))
            # h0 is (n*300 x 300 x d)
            h1 = lrelu(self.d_bn1(e2e(h0, self.df_dim*2,k_h=self.graph_size[0], name='d_h1_conv')))
            # h1 is (n*300 x 300 x d)
            h2 = lrelu(self.d_bn2(e2n(h1, self.df_dim*2, k_h=self.graph_size[0],name='d_h2_conv')))
            # h2 is (n*300x 1 x d)
            h3 = lrelu(self.d_bn3(n2g(h2, self.df_dim*2,k_h=self.graph_size[0],  name='d_h3_conv')))
            # h3 is (n*1x1xd)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            # h4 is (n*d)
            return tf.nn.sigmoid(h4), h4

    def generator(self, graph, y=None):
        with tf.variable_scope("generator") as scope:      
            # graph is (n*300 x 300 x 1)
            e1 = self.g_bn_e1(e2e(lrelu(graph), self.gf_dim, k_h=self.graph_size[0],name='g_e1_conv'))
            # e1 is (n*300 x 300*d )
            e2 = self.g_bn_e2(e2e(lrelu(e1), self.gf_dim*2, k_h=self.graph_size[0],name='g_e2_conv'))
            e2_=tf.nn.dropout(e2,0.5)
            # e2 is (n*300 x 300*d )
            e3 = self.g_bn_e3(e2n(lrelu(e2_), self.gf_dim*2,k_h=self.graph_size[0], name='g_e3_conv'))
            # e3 is (n*300 x 1*d )
           # e4 = self.g_bn_e4(n2g(lrelu(e3), self.gf_dim*2, name='g_e4_conv'))
            # e4 is (n*1 x 1*d )


            #self.d1, self.d1_w, self.d1_b = de_n2g(tf.nn.relu(e4),
             #   [self.batch_size, 300, 1, self.gf_dim*2], name='g_d1', with_w=True)
            #d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            #d1 = tf.concat([d1, e3], 3)
             #d1 is (300 x 1 )
            self.d2, self.d2_w, self.d2_b = de_e2n(tf.nn.relu(e3),
                [self.batch_size, self.graph_size[0], self.graph_size[0], self.gf_dim*2],k_h=self.graph_size[0], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e2], 3)
            # d2 is (300 x 300 )
            self.d3, self.d3_w, self.d3_b = de_e2e(tf.nn.relu(d2),
                [self.batch_size,self.graph_size[0], self.graph_size[0], int(self.gf_dim)],k_h=self.graph_size[0], name='g_d3', with_w=True)
            d3 = self.g_bn_d3(self.d3)
            d3 = tf.concat([d3, e1], 3)
            # d3 is (300 x 300 )

            self.d4, self.d4_w, self.d4_b = de_e2e(tf.nn.relu(d3),
                [self.batch_size, self.graph_size[0], self.graph_size[0], self.output_c_dim],k_h=self.graph_size[0], name='g_d4', with_w=True)


            return  tf.add(tf.nn.relu(self.d4),graph)

    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
   
    def test_auth(self, args,filename):
        score=[]      
        gen_data=[]
        def mse(A,B):
          MSE=[]
          for i in range(len(A)):
            MSE.append(mean_squared_error(A[i],B[i]))
          return np.mean(MSE)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load testing input
        print("Loading testing graphs ...")
        sample_graphs_all =load_data_test(args.test_dir,filename,self.graph_size[0])
        sample_graphs = [sample_graphs_all[i:i+self.batch_size]
                         for i in xrange(0, len(sample_graphs_all), self.batch_size)]
        sample_graphs = np.array(sample_graphs)  
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for i, sample_graph in enumerate(sample_graphs):
            idx = i+1
            print("sampling graph ", idx)
            samples = self.sess.run(
                self.fake_B,
                feed_dict={self.real_data: sample_graphs[i]}
            )
            
            label = self.sess.run(
                self.real_B,
                feed_dict={self.real_data: sample_graphs[i]}
            )
            if i==0: gen_data=samples
            if i>0: gen_data=np.concatenate((gen_data,samples),axis=0)

        np.save(self.test_dir+filename.split('.')[0]+'_gen50.npy',gen_data)
        
        
    def test(self, args):        
        score=[]      
        gen_data=[]

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load testing input
        print("Loading testing graphs ...")
        sample_graphs_all =load_data_test(self.graph_size[0],self.dataset)
        sample_graphs = [sample_graphs_all[i:i+self.batch_size]
                         for i in xrange(0, len(sample_graphs_all), self.batch_size)]
        sample_graphs = np.array(sample_graphs)  
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for i, sample_graph in enumerate(sample_graphs):
            idx = i+1
            print("sampling graph ", idx)
            samples = self.sess.run(
                self.fake_B,
                feed_dict={self.real_data: sample_graphs[i]}
            )
            
            label = self.sess.run(
                self.real_B,
                feed_dict={self.real_data: sample_graphs[i]}
            )
            if i==0: gen_data=samples
            if i>0: gen_data=np.concatenate((gen_data,samples),axis=0)
        for i in range(gen_data.shape[0]):
            for j in range(gen_data.shape[1]):
                for k in range(gen_data.shape[2]):
                    gen_data[i,j,k,0]=round(gen_data[i,j,k,0])
        
        if self.dataset=='scale-free':
           np.save(self.test_dir+'scale_gen'+str(self.graph_size[0])+'.npy',gen_data)
        if self.dataset=='poisson-random':
           np.save(self.test_dir+'poisson_gen'+str(self.graph_size[0])+'.npy',gen_data)
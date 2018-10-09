import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from utils_ import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
  
    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)



def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=3, d_w=3, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
def e2e(input_,output_dim,k_h=50, d_h=1, d_w=1, stddev=0.02,
           name="e2e"):
    with tf.variable_scope(name):
        w1 = tf.get_variable('w1', [k_h, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1 = tf.nn.conv2d(input_, w1[0:1,:,:,:], strides=[1, d_h, d_w, 1], padding='VALID')
        biases1 = tf.get_variable('biases1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.reshape(tf.nn.bias_add(conv1, biases1), conv1.get_shape())
        w2 = tf.get_variable('w2', [k_h,k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv2 = tf.nn.conv2d(input_, w2[:,0:1,:,:], strides=[1, d_h, d_w, 1], padding='VALID')
        biases2 = tf.get_variable('biases2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.reshape(tf.nn.bias_add(conv2, biases2), conv2.get_shape())
        m1 = tf.tile(conv1,[1,1,k_h,1])
        m2 = tf.tile(conv2,[1,k_h,1,1])
        conv = tf.add(m1, m2)
        return conv
    
def e2n(input_,output_dim,k_h=50, d_h=1, d_w=1, stddev=0.02,
           name="e2n"):
     with tf.variable_scope(name):
        w = tf.get_variable('w', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
    
def n2g(input_,output_dim,k_h=50, d_h=1, d_w=1, stddev=0.02,
           name="e2n"):
     with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h,1, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def de_n2g(input_, output_shape,
             k_h=50, d_h=1, d_w=1, stddev=0.02,
             name="de_n2g", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h,1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv    

def de_e2n(input_, output_shape,
             k_h=50, d_h=1, d_w=1, stddev=0.02,
             name="de_n2g", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w, biases
        else:
            return deconv  
         
def de_e2e(input_, output_shape,
             k_h=50, d_h=1, d_w=1, stddev=0.02,
             name="de_n2g", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        input_1=tf.reshape(tf.reduce_sum(input_,axis=1),(int(input_.shape[0]),k_h,1,int(input_.shape[3]))) 
        input_2=tf.reshape(tf.reduce_sum(input_,axis=2),(int(input_.shape[0]),1,k_h,int(input_.shape[3]))) 
        
        w1 = tf.get_variable('w1', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv1 = tf.nn.conv2d_transpose(input_1, w1, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')       
        biases1 = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, biases1), deconv1.get_shape())

        w2 = tf.get_variable('w2', [k_h,1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv2 = tf.nn.conv2d_transpose(input_2, w2, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases2 = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))        
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases2), deconv2.get_shape())
        
        deconv=tf.add(deconv1,deconv2)/2
        if with_w:
            return deconv, w1, biases1
        else:
            return deconv  
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=3, d_w=3, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

       

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def linear_mask(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear_mask"):
        matrix = np.ones((shape[1], output_size[1])).astype('float32')
        mask_ = np.loadtxt('mask.csv',delimiter=',')
        mask = mask_.astype('float32')
        output=tf.matmul(input_, matrix*mask) 
    return tf.reshape(output,[shape[0],54,1,1])

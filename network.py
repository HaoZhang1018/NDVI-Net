# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup_NDVI, 
  input_setup_HRVI, 
  imsave,
  merge,
  sobel_gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size_NDVI=25,
               image_size_HRVI=100,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size_NDVI =  image_size_NDVI
    self.image_size_HRVI =  image_size_HRVI
    self.image_size_Label = image_size_HRVI
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('NDVI_input'):
        self.images_NDVI = tf.placeholder(tf.float32, [None, self.image_size_NDVI, self.image_size_NDVI, self.c_dim], name='images_NDVI')
    with tf.name_scope('HRVI_input'):
        self.images_HRVI = tf.placeholder(tf.float32, [None, self.image_size_HRVI, self.image_size_HRVI, self.c_dim], name='images_HRVI')
    with tf.name_scope('Label_input'):
        self.images_Label = tf.placeholder(tf.float32, [None, self.image_size_Label, self.image_size_Label, self.c_dim], name='images_Label')

    with tf.name_scope('input'):
        self.input_image_NDVI = self.images_NDVI 
        self.input_image_HRVI = self.images_HRVI


    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_image_NDVI,self.input_image_HRVI)

    with tf.name_scope('Gradient'): 
        self.fusion_image_gradient_x, self.fusion_image_gradient_y=sobel_gradient(self.fusion_image)
        self.images_Label_gradient_x, self.images_Label_gradient_y=sobel_gradient(self.images_Label)


    with tf.name_scope('g_loss'):
        self.g_loss_int  = tf.reduce_mean(tf.abs(self.fusion_image - self.images_Label))
        self.g_loss_grad = tf.reduce_mean(tf.abs(self.fusion_image_gradient_x-self.images_Label_gradient_x))+ tf.reduce_mean(tf.abs(self.fusion_image_gradient_y-self.images_Label_gradient_y))
        


        self.g_loss_total=100*(1*self.g_loss_int+5*self.g_loss_grad)

        tf.summary.scalar('g_loss_int',self.g_loss_int)
        tf.summary.scalar('g_loss_grad',self.g_loss_grad)        
        tf.summary.scalar('g_loss_total',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=30)
    with tf.name_scope('image'):
        tf.summary.image('input_image_NDVI',tf.expand_dims(self.input_image_NDVI[1,:,:,:],0))  
        tf.summary.image('input_image_HRVI',tf.expand_dims(self.input_image_HRVI[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))
        tf.summary.image('images_Label',tf.expand_dims(self.images_Label[1,:,:,:],0)) 
                 
        tf.summary.image('fusion_image_gradient_x',tf.expand_dims(self.fusion_image_gradient_x[1,:,:,:],0))  
        tf.summary.image('fusion_image_gradient_y',tf.expand_dims(self.fusion_image_gradient_y[1,:,:,:],0))
        tf.summary.image('images_Label_gradient_x',tf.expand_dims(self.images_Label_gradient_x[1,:,:,:],0)) 
        tf.summary.image('images_Label_gradient_y',tf.expand_dims(self.images_Label_gradient_y[1,:,:,:],0))  


    
  def train(self, config):
    if config.is_train:
      input_setup_NDVI(self.sess, config,"Train_NDVI")
      input_setup_HRVI(self.sess,config,"Train_HRVI")
      input_setup_HRVI(self.sess,config,"Train_Label")
    if config.is_train:     
      data_dir_NDVI = os.path.join('./{}'.format(config.checkpoint_dir), "Train_NDVI","train.h5")
      data_dir_HRVI = os.path.join('./{}'.format(config.checkpoint_dir), "Train_HRVI","train.h5")
      data_dir_Label = os.path.join('./{}'.format(config.checkpoint_dir), "Train_Label","train.h5")
      
    train_data_NDVI= read_data(data_dir_NDVI)
    train_data_HRVI= read_data(data_dir_HRVI)
    train_data_Label= read_data(data_dir_Label)
    
    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
    self.summary_op = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)    
    tf.initialize_all_variables().run()    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_NDVI) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_NDVI = train_data_NDVI[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_HRVI = train_data_HRVI[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_Label = train_data_Label[idx*config.batch_size : (idx+1)*config.batch_size]
          counter += 1
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_NDVI: batch_images_NDVI, self.images_HRVI: batch_images_HRVI,self.images_Label: batch_images_Label})
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g))
        self.save(config.checkpoint_dir, ep)

  def fusion_model(self,img_NDVI,img_HRVI):
######################################################### 
####################  Deconv  ###########################
######################################################### 
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1_Deconv'):
            weights=tf.get_variable("w1_Deconv",[5,5,16,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            conv1_NDVI=tf.nn.conv2d_transpose(img_NDVI, weights, [32,50,50,16],strides=[1,2,2,1], padding='SAME')  
        with tf.variable_scope('layer2_Deconv'):
            weights=tf.get_variable("w2_Deconv",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            conv2_NDVI= tf.nn.conv2d_transpose(conv1_NDVI, weights, [32,100,100,1],strides=[1,2,2,1], padding='SAME') 
   
    
##########  Multi_Scale with Channel Attention ###############    
######################################################### 
#################### NDVI Layer 1 ###########################
######################################################### 
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('layer1_NDVI_3x3'):
            weights=tf.get_variable("w1_NDVI_3x3",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_3x3=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_3x3 = lrelu(conv1_NDVI_3x3) 
        with tf.variable_scope('layer1_NDVI_5x5'):
            weights=tf.get_variable("w1_NDVI_5x5",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_5x5=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_5x5 = lrelu(conv1_NDVI_5x5) 
        with tf.variable_scope('layer1_NDVI_7x7'):
            weights=tf.get_variable("w1_NDVI_7x7",[7,7,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_7x7=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_7x7 = lrelu(conv1_NDVI_7x7)
        conv1_NDVI_cat=tf.concat([conv1_NDVI_3x3,conv1_NDVI_5x5,conv1_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_NDVI_1_max=tf.reduce_max(conv1_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_1_mean=tf.reduce_mean(conv1_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_NDVI_max_1'):
            weights=tf.get_variable("w1_NDVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_max_1 = tf.nn.relu(conv1_NDVI_max_1)
        with tf.variable_scope('layer1_NDVI_mean_1'):
            weights=tf.get_variable("w1_NDVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_mean_1 = tf.nn.relu(conv1_NDVI_mean_1)            

        with tf.variable_scope('layer1_NDVI_max_2'):
            weights=tf.get_variable("w1_NDVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_max_2=tf.nn.conv2d(conv1_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_NDVI_mean_2'):
            weights=tf.get_variable("w1_NDVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_NDVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_NDVI_mean_2=tf.nn.conv2d(conv1_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_NDVI_atten_map= tf.nn.sigmoid(conv1_NDVI_max_2+conv1_NDVI_mean_2)    
        conv1_NDVI_atten_out= conv1_NDVI_cat*conv1_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 1 ###########################
######################################################### 
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('layer1_HRVI_3x3'):
            weights=tf.get_variable("w1_HRVI_3x3",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_3x3=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_3x3 = lrelu(conv1_HRVI_3x3) 
        with tf.variable_scope('layer1_HRVI_5x5'):
            weights=tf.get_variable("w1_HRVI_5x5",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_5x5=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_5x5 = lrelu(conv1_HRVI_5x5) 
        with tf.variable_scope('layer1_HRVI_7x7'):
            weights=tf.get_variable("w1_HRVI_7x7",[7,7,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_7x7=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_7x7 = lrelu(conv1_HRVI_7x7)
        conv1_HRVI_cat=tf.concat([conv1_HRVI_3x3,conv1_HRVI_5x5,conv1_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_HRVI_1_max=tf.reduce_max(conv1_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_1_mean=tf.reduce_mean(conv1_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_HRVI_max_1'):
            weights=tf.get_variable("w1_HRVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_max_1 = tf.nn.relu(conv1_HRVI_max_1)
        with tf.variable_scope('layer1_HRVI_mean_1'):
            weights=tf.get_variable("w1_HRVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_mean_1 = tf.nn.relu(conv1_HRVI_mean_1)            

        with tf.variable_scope('layer1_HRVI_max_2'):
            weights=tf.get_variable("w1_HRVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_max_2=tf.nn.conv2d(conv1_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_HRVI_mean_2'):
            weights=tf.get_variable("w1_HRVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_HRVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv1_HRVI_mean_2=tf.nn.conv2d(conv1_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_HRVI_atten_map= tf.nn.sigmoid(conv1_HRVI_max_2+conv1_HRVI_mean_2)    
        conv1_HRVI_atten_out= conv1_HRVI_cat*conv1_HRVI_atten_map    
        
                
######################################################### 
#################### NDVI Layer 2 #######################
#########################################################  
##################  Multi_Scale 2 #######################     
        NDVI_dense_HR_t2=tf.concat([conv1_NDVI_atten_out,conv1_HRVI_atten_out],axis=-1)
        with tf.variable_scope('layer2_NDVI_3x3'):
            weights=tf.get_variable("w2_NDVI_3x3",[3,3,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_3x3 = lrelu(conv2_NDVI_3x3) 
        with tf.variable_scope('layer2_NDVI_5x5'):
            weights=tf.get_variable("w2_NDVI_5x5",[5,5,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_5x5 = lrelu(conv2_NDVI_5x5) 
        with tf.variable_scope('layer2_NDVI_7x7'):
            weights=tf.get_variable("w2_NDVI_7x7",[7,7,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_7x7 = lrelu(conv2_NDVI_7x7)
        conv2_NDVI_cat=tf.concat([conv2_NDVI_3x3,conv2_NDVI_5x5,conv2_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 2 ######################   
        CAttent_NDVI_2_max=tf.reduce_max(conv2_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_2_mean=tf.reduce_mean(conv2_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_NDVI_max_1'):
            weights=tf.get_variable("w2_NDVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_max_1 = tf.nn.relu(conv2_NDVI_max_1)
        with tf.variable_scope('layer2_NDVI_mean_1'):
            weights=tf.get_variable("w2_NDVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_mean_1 = tf.nn.relu(conv2_NDVI_mean_1)            

        with tf.variable_scope('layer2_NDVI_max_2'):
            weights=tf.get_variable("w2_NDVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_max_2=tf.nn.conv2d(conv2_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_NDVI_mean_2'):
            weights=tf.get_variable("w2_NDVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_NDVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_NDVI_mean_2=tf.nn.conv2d(conv2_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_NDVI_atten_map= tf.nn.sigmoid(conv2_NDVI_max_2+conv2_NDVI_mean_2)    
        conv2_NDVI_atten_out= conv2_NDVI_cat*conv2_NDVI_atten_map     
            

######################################################### 
#################### HRVI Layer 2 #######################
#########################################################  
##################  Multi_Scale 2 #######################     
        with tf.variable_scope('layer2_HRVI_3x3'):
            weights=tf.get_variable("w2_HRVI_3x3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_3x3=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_3x3 = lrelu(conv2_HRVI_3x3) 
        with tf.variable_scope('layer2_HRVI_5x5'):
            weights=tf.get_variable("w2_HRVI_5x5",[5,5,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_5x5=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_5x5 = lrelu(conv2_HRVI_5x5) 
        with tf.variable_scope('layer2_HRVI_7x7'):
            weights=tf.get_variable("w2_HRVI_7x7",[7,7,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_7x7=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_7x7 = lrelu(conv2_HRVI_7x7)
        conv2_HRVI_cat=tf.concat([conv2_HRVI_3x3,conv2_HRVI_5x5,conv2_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 2 ######################   
        CAttent_HRVI_2_max=tf.reduce_max(conv2_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_2_mean=tf.reduce_mean(conv2_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_HRVI_max_1'):
            weights=tf.get_variable("w2_HRVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_max_1 = tf.nn.relu(conv2_HRVI_max_1)
        with tf.variable_scope('layer2_HRVI_mean_1'):
            weights=tf.get_variable("w2_HRVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_mean_1 = tf.nn.relu(conv2_HRVI_mean_1)            

        with tf.variable_scope('layer2_HRVI_max_2'):
            weights=tf.get_variable("w2_HRVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_max_2=tf.nn.conv2d(conv2_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_HRVI_mean_2'):
            weights=tf.get_variable("w2_HRVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_HRVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv2_HRVI_mean_2=tf.nn.conv2d(conv2_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_HRVI_atten_map= tf.nn.sigmoid(conv2_HRVI_max_2+conv2_HRVI_mean_2)    
        conv2_HRVI_atten_out= conv2_HRVI_cat*conv2_HRVI_atten_map 
            
######################################################### 
#################### NDVI Layer 3 #######################
#########################################################  
##################  Multi_Scale 3 ####################### 
        NDVI_dense_HR_t3=tf.concat([conv1_NDVI_atten_out,conv2_NDVI_atten_out,conv2_HRVI_atten_out],axis=-1)    
           
        with tf.variable_scope('layer3_NDVI_3x3'):
            weights=tf.get_variable("w3_NDVI_3x3",[3,3,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_3x3 = lrelu(conv3_NDVI_3x3) 
        with tf.variable_scope('layer3_NDVI_5x5'):
            weights=tf.get_variable("w3_NDVI_5x5",[5,5,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_5x5 = lrelu(conv3_NDVI_5x5) 
        with tf.variable_scope('layer3_NDVI_7x7'):
            weights=tf.get_variable("w3_NDVI_7x7",[7,7,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_7x7 = lrelu(conv3_NDVI_7x7)
        conv3_NDVI_cat=tf.concat([conv3_NDVI_3x3,conv3_NDVI_5x5,conv3_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_NDVI_3_max=tf.reduce_max(conv3_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_3_mean=tf.reduce_mean(conv3_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer3_NDVI_max_1'):
            weights=tf.get_variable("w3_NDVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_max_1 = tf.nn.relu(conv3_NDVI_max_1)
        with tf.variable_scope('layer3_NDVI_mean_1'):
            weights=tf.get_variable("w3_NDVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_3_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_mean_1 = tf.nn.relu(conv3_NDVI_mean_1)            

        with tf.variable_scope('layer3_NDVI_max_2'):
            weights=tf.get_variable("w3_NDVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_max_2=tf.nn.conv2d(conv3_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer3_NDVI_mean_2'):
            weights=tf.get_variable("w3_NDVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_NDVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv3_NDVI_mean_2=tf.nn.conv2d(conv3_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv3_NDVI_atten_map= tf.nn.sigmoid(conv3_NDVI_max_2+conv3_NDVI_mean_2)    
        conv3_NDVI_atten_out= conv3_NDVI_cat*conv3_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 3 #######################
#########################################################  
##################  Multi_Scale 3 #######################
        HRVI_dense_t3=tf.concat([conv1_HRVI_atten_out,conv2_HRVI_atten_out],axis=-1)              
        with tf.variable_scope('layer3_HRVI_3x3'):
            weights=tf.get_variable("w3_HRVI_3x3",[3,3,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_3x3=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_3x3 = lrelu(conv3_HRVI_3x3) 
        with tf.variable_scope('layer3_HRVI_5x5'):
            weights=tf.get_variable("w3_HRVI_5x5",[5,5,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_5x5=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_5x5 = lrelu(conv3_HRVI_5x5) 
        with tf.variable_scope('layer3_HRVI_7x7'):
            weights=tf.get_variable("w3_HRVI_7x7",[7,7,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_7x7=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_7x7 = lrelu(conv3_HRVI_7x7)
        conv3_HRVI_cat=tf.concat([conv3_HRVI_3x3,conv3_HRVI_5x5,conv3_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_HRVI_3_max=tf.reduce_max(conv3_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_3_mean=tf.reduce_mean(conv3_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer3_HRVI_max_1'):
            weights=tf.get_variable("w3_HRVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_max_1 = tf.nn.relu(conv3_HRVI_max_1)
        with tf.variable_scope('layer3_HRVI_mean_1'):
            weights=tf.get_variable("w3_HRVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_3_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_mean_1 = tf.nn.relu(conv3_HRVI_mean_1)            

        with tf.variable_scope('layer3_HRVI_max_2'):
            weights=tf.get_variable("w3_HRVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_max_2=tf.nn.conv2d(conv3_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer3_HRVI_mean_2'):
            weights=tf.get_variable("w3_HRVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_HRVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv3_HRVI_mean_2=tf.nn.conv2d(conv3_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv3_HRVI_atten_map= tf.nn.sigmoid(conv3_HRVI_max_2+conv3_HRVI_mean_2)    
        conv3_HRVI_atten_out= conv3_HRVI_cat*conv3_HRVI_atten_map    


######################################################### 
#################### NDVI Layer 4 #######################
#########################################################  
##################  Multi_Scale 4 ####################### 
        NDVI_dense_HR_t4=tf.concat([conv1_NDVI_atten_out,conv2_NDVI_atten_out,conv3_NDVI_atten_out,conv3_HRVI_atten_out],axis=-1)         
        with tf.variable_scope('layer4_NDVI_3x3'):
            weights=tf.get_variable("w4_NDVI_3x3",[3,3,192,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_3x3 = lrelu(conv4_NDVI_3x3) 
        with tf.variable_scope('layer4_NDVI_5x5'):
            weights=tf.get_variable("w4_NDVI_5x5",[5,5,192,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_5x5 = lrelu(conv4_NDVI_5x5) 
        with tf.variable_scope('layer4_NDVI_7x7'):
            weights=tf.get_variable("w4_NDVI_7x7",[7,7,192,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_7x7 = lrelu(conv4_NDVI_7x7)
        conv4_NDVI_cat=tf.concat([conv4_NDVI_3x3,conv4_NDVI_5x5,conv4_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 4 ######################   
        CAttent_NDVI_4_max=tf.reduce_max(conv4_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_4_mean=tf.reduce_mean(conv4_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer4_NDVI_max_1'):
            weights=tf.get_variable("w4_NDVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_4_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_max_1 = tf.nn.relu(conv4_NDVI_max_1)
        with tf.variable_scope('layer4_NDVI_mean_1'):
            weights=tf.get_variable("w4_NDVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_4_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_mean_1 = tf.nn.relu(conv4_NDVI_mean_1)            

        with tf.variable_scope('layer4_NDVI_max_2'):
            weights=tf.get_variable("w4_NDVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_max_2=tf.nn.conv2d(conv4_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer4_NDVI_mean_2'):
            weights=tf.get_variable("w4_NDVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_NDVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv4_NDVI_mean_2=tf.nn.conv2d(conv4_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv4_NDVI_atten_map= tf.nn.sigmoid(conv4_NDVI_max_2+conv4_NDVI_mean_2)    
        conv4_NDVI_atten_out= conv4_NDVI_cat*conv4_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 4 #######################
#########################################################  
##################  Multi_Scale 4 #######################
        HRVI_dense_t4=tf.concat([conv1_HRVI_atten_out,conv2_HRVI_atten_out,conv3_HRVI_atten_out],axis=-1)              
        with tf.variable_scope('layer4_HRVI_3x3'):
            weights=tf.get_variable("w4_HRVI_3x3",[3,3,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_3x3",[16],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_3x3=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_3x3 = lrelu(conv4_HRVI_3x3) 
        with tf.variable_scope('layer4_HRVI_5x5'):
            weights=tf.get_variable("w4_HRVI_5x5",[5,5,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_5x5",[16],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_5x5=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_5x5 = lrelu(conv4_HRVI_5x5) 
        with tf.variable_scope('layer4_HRVI_7x7'):
            weights=tf.get_variable("w4_HRVI_7x7",[7,7,144,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_7x7",[16],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_7x7=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_7x7 = lrelu(conv4_HRVI_7x7)
        conv4_HRVI_cat=tf.concat([conv4_HRVI_3x3,conv4_HRVI_5x5,conv4_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_HRVI_4_max=tf.reduce_max(conv4_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_4_mean=tf.reduce_mean(conv4_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer4_HRVI_max_1'):
            weights=tf.get_variable("w4_HRVI_max_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_max_1",[12],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_4_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_max_1 = tf.nn.relu(conv4_HRVI_max_1)
        with tf.variable_scope('layer4_HRVI_mean_1'):
            weights=tf.get_variable("w4_HRVI_mean_1",[1,1,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_mean_1",[12],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_4_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_mean_1 = tf.nn.relu(conv4_HRVI_mean_1)            

        with tf.variable_scope('layer4_HRVI_max_2'):
            weights=tf.get_variable("w4_HRVI_max_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_max_2",[48],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_max_2=tf.nn.conv2d(conv4_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer4_HRVI_mean_2'):
            weights=tf.get_variable("w4_HRVI_mean_2",[1,1,12,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_HRVI_mean_2",[48],initializer=tf.constant_initializer(0.0))
            conv4_HRVI_mean_2=tf.nn.conv2d(conv4_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv4_HRVI_atten_map= tf.nn.sigmoid(conv4_HRVI_max_2+conv4_HRVI_mean_2)    
        conv4_HRVI_atten_out= conv4_HRVI_cat*conv4_HRVI_atten_map       


################ Spatial Attention ###################### 
        NDVI_HRVI_cat=tf.concat([conv4_NDVI_atten_out,conv4_HRVI_atten_out],axis=-1)      
######################################################### 
####################  Layer 5 ###########################
#########################################################  
####################  Conv 5  ###########################            
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[5,5,96,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5",[16],initializer=tf.constant_initializer(0.0))
            conv5=tf.nn.conv2d(NDVI_HRVI_cat, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5 = lrelu(conv5)                                         
#################  Spatial Attention 4 ####################   
        SAttent_5_max=tf.reduce_max(conv5, axis=3, keepdims=True)
        SAttent_5_mean=tf.reduce_mean(conv5, axis=3, keepdims=True)
        SAttent_5_cat_mean_max=tf.concat([SAttent_5_max,SAttent_5_mean],axis=-1)        
        with tf.variable_scope('layer5_atten_map'):
            weights=tf.get_variable("w5_atten_map",[7,7,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv5_atten_map=tf.nn.conv2d(SAttent_5_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_atten_map = tf.nn.sigmoid(conv5_atten_map)  
        conv5_atten_out= conv5*conv5_atten_map    

######################################################### 
####################  Layer 6 ###########################
#########################################################  
####################  Conv 6  ###########################            
        with tf.variable_scope('layer6'):
            weights=tf.get_variable("w6",[5,5,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b6",[16],initializer=tf.constant_initializer(0.0))
            conv6=tf.nn.conv2d(conv5_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6 = lrelu(conv6)                                         
#################  Spatial Attention 6 ####################   
        SAttent_6_max=tf.reduce_max(conv6, axis=3, keepdims=True)
        SAttent_6_mean=tf.reduce_mean(conv6, axis=3, keepdims=True)
        SAttent_6_cat_mean_max=tf.concat([SAttent_6_max,SAttent_6_mean],axis=-1)
        
        with tf.variable_scope('layer6_atten_map'):
            weights=tf.get_variable("w6_atten_map",[7,7,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b6_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv6_atten_map=tf.nn.conv2d(SAttent_6_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_atten_map = tf.nn.sigmoid(conv6_atten_map)  
        conv6_atten_out= conv6*conv6_atten_map    

######################################################### 
####################  Layer 7 ###########################
#########################################################  
        sa_dense_t7=tf.concat([conv5_atten_out,conv6_atten_out],axis=-1)
####################  Conv 6  ###########################            
        with tf.variable_scope('layer7'):
            weights=tf.get_variable("w7",[5,5,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b7",[16],initializer=tf.constant_initializer(0.0))
            conv7=tf.nn.conv2d(sa_dense_t7, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7 = lrelu(conv7)                                         
#################  Spatial Attention 6 ####################   
        SAttent_7_max=tf.reduce_max(conv7, axis=3, keepdims=True)
        SAttent_7_mean=tf.reduce_mean(conv7, axis=3, keepdims=True)
        SAttent_7_cat_mean_max=tf.concat([SAttent_7_max,SAttent_7_mean],axis=-1)
        
        with tf.variable_scope('layer7_atten_map'):
            weights=tf.get_variable("w7_atten_map",[7,7,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b7_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv7_atten_map=tf.nn.conv2d(SAttent_7_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_atten_map = tf.nn.sigmoid(conv7_atten_map)  
        conv7_atten_out= conv7*conv7_atten_map  

######################################################### 
####################  Layer 8 ###########################
#########################################################  
        sa_dense_t8=tf.concat([conv5_atten_out,conv6_atten_out,conv7_atten_out],axis=-1)
####################  Conv 8  ###########################            
        with tf.variable_scope('layer8'):
            weights=tf.get_variable("w8",[5,5,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b8",[16],initializer=tf.constant_initializer(0.0))
            conv8=tf.nn.conv2d(sa_dense_t8, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8 = lrelu(conv8)                                         
#################  Spatial Attention 8 ####################   
        SAttent_8_max=tf.reduce_max(conv8, axis=3, keepdims=True)
        SAttent_8_mean=tf.reduce_mean(conv8, axis=3, keepdims=True)
        SAttent_8_cat_mean_max=tf.concat([SAttent_8_max,SAttent_8_mean],axis=-1)
        
        with tf.variable_scope('layer8_atten_map'):
            weights=tf.get_variable("w8_atten_map",[7,7,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b8_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv8_atten_map=tf.nn.conv2d(SAttent_8_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_atten_map = tf.nn.sigmoid(conv8_atten_map)  
        conv8_atten_out= conv8*conv8_atten_map  
            
################ Reconstruction ###################### 

######################################################### 
####################  Layer 9 ###########################
######################################################### 
        with tf.variable_scope('layer9'):
            weights=tf.get_variable("w9",[5,5,16,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b9",[4],initializer=tf.constant_initializer(0.0))
            conv9=tf.nn.conv2d(conv8_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9 =lrelu(conv9)  
######################################################### 
####################  Layer 8 ###########################
######################################################### 
        with tf.variable_scope('layer10'):
            weights=tf.get_variable("w10",[5,5,4,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b10",[1],initializer=tf.constant_initializer(0.0))
            conv10=tf.nn.conv2d(conv9, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv10 =tf.nn.tanh(conv10)  
    return conv10
    

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s" % ("CGAN")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False

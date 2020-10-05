# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image.astype(np.uint8))
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_NDVI,img_HRVI):
    with tf.variable_scope('fusion_model'):
######################################################### 
####################  Deconv  ###########################
######################################################### 
        with tf.variable_scope('layer1_Deconv'):
            weights=tf.get_variable("w1_Deconv",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_Deconv/w1_Deconv')))
            conv1_NDVI=tf.nn.conv2d_transpose(img_NDVI, weights, [1,200,200,16],strides=[1,2,2,1], padding='SAME')  
        with tf.variable_scope('layer2_Deconv'):
            weights=tf.get_variable("w2_Deconv",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_Deconv/w2_Deconv')))
            conv2_NDVI= tf.nn.conv2d_transpose(conv1_NDVI, weights, [1,400,400,1],strides=[1,2,2,1], padding='SAME') 
   
    
##########  Multi_Scale with Channel Attention ###############    
######################################################### 
#################### NDVI Layer 1 ###########################
######################################################### 
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('layer1_NDVI_3x3'):
            weights=tf.get_variable("w1_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_3x3/w1_NDVI_3x3')))
            bias=tf.get_variable("b1_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_3x3/b1_NDVI_3x3')))
            conv1_NDVI_3x3=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_3x3 = lrelu(conv1_NDVI_3x3) 
        with tf.variable_scope('layer1_NDVI_5x5'):
            weights=tf.get_variable("w1_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_5x5/w1_NDVI_5x5')))
            bias=tf.get_variable("b1_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_5x5/b1_NDVI_5x5')))
            conv1_NDVI_5x5=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_5x5 = lrelu(conv1_NDVI_5x5) 
        with tf.variable_scope('layer1_NDVI_7x7'):
            weights=tf.get_variable("w1_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_7x7/w1_NDVI_7x7')))
            bias=tf.get_variable("b1_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_7x7/b1_NDVI_7x7')))
            conv1_NDVI_7x7=tf.nn.conv2d(conv2_NDVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_7x7 = lrelu(conv1_NDVI_7x7)
        conv1_NDVI_cat=tf.concat([conv1_NDVI_3x3,conv1_NDVI_5x5,conv1_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_NDVI_1_max=tf.reduce_max(conv1_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_1_mean=tf.reduce_mean(conv1_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_NDVI_max_1'):
            weights=tf.get_variable("w1_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_max_1/w1_NDVI_max_1')))
            bias=tf.get_variable("b1_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_max_1/b1_NDVI_max_1')))
            conv1_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_max_1 = tf.nn.relu(conv1_NDVI_max_1)
        with tf.variable_scope('layer1_NDVI_mean_1'):
            weights=tf.get_variable("w1_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_mean_1/w1_NDVI_mean_1')))
            bias=tf.get_variable("b1_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_mean_1/b1_NDVI_mean_1')))
            conv1_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_NDVI_mean_1 = tf.nn.relu(conv1_NDVI_mean_1)            

        with tf.variable_scope('layer1_NDVI_max_2'):
            weights=tf.get_variable("w1_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_max_2/w1_NDVI_max_2')))
            bias=tf.get_variable("b1_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_max_2/b1_NDVI_max_2')))
            conv1_NDVI_max_2=tf.nn.conv2d(conv1_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_NDVI_mean_2'):
            weights=tf.get_variable("w1_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_mean_2/w1_NDVI_mean_2')))
            bias=tf.get_variable("b1_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_NDVI_mean_2/b1_NDVI_mean_2')))
            conv1_NDVI_mean_2=tf.nn.conv2d(conv1_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_NDVI_atten_map= tf.nn.sigmoid(conv1_NDVI_max_2+conv1_NDVI_mean_2)    
        conv1_NDVI_atten_out= conv1_NDVI_cat*conv1_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 1 ###########################
######################################################### 
##################  Multi_Scale 1 #######################       
        with tf.variable_scope('layer1_HRVI_3x3'):
            weights=tf.get_variable("w1_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_3x3/w1_HRVI_3x3')))
            bias=tf.get_variable("b1_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_3x3/b1_HRVI_3x3')))
            conv1_HRVI_3x3=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_3x3 = lrelu(conv1_HRVI_3x3) 
        with tf.variable_scope('layer1_HRVI_5x5'):
            weights=tf.get_variable("w1_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_5x5/w1_HRVI_5x5')))
            bias=tf.get_variable("b1_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_5x5/b1_HRVI_5x5')))
            conv1_HRVI_5x5=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_5x5 = lrelu(conv1_HRVI_5x5) 
        with tf.variable_scope('layer1_HRVI_7x7'):
            weights=tf.get_variable("w1_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_7x7/w1_HRVI_7x7')))
            bias=tf.get_variable("b1_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_7x7/b1_HRVI_7x7')))
            conv1_HRVI_7x7=tf.nn.conv2d(img_HRVI, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_7x7 = lrelu(conv1_HRVI_7x7)
        conv1_HRVI_cat=tf.concat([conv1_HRVI_3x3,conv1_HRVI_5x5,conv1_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 1 #####################   
        CAttent_HRVI_1_max=tf.reduce_max(conv1_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_1_mean=tf.reduce_mean(conv1_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer1_HRVI_max_1'):
            weights=tf.get_variable("w1_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_max_1/w1_HRVI_max_1')))
            bias=tf.get_variable("b1_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_max_1/b1_HRVI_max_1')))
            conv1_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_max_1 = tf.nn.relu(conv1_HRVI_max_1)
        with tf.variable_scope('layer1_HRVI_mean_1'):
            weights=tf.get_variable("w1_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_mean_1/w1_HRVI_mean_1')))
            bias=tf.get_variable("b1_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_mean_1/b1_HRVI_mean_1')))
            conv1_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_1_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_HRVI_mean_1 = tf.nn.relu(conv1_HRVI_mean_1)            

        with tf.variable_scope('layer1_HRVI_max_2'):
            weights=tf.get_variable("w1_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_max_2/w1_HRVI_max_2')))
            bias=tf.get_variable("b1_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_max_2/b1_HRVI_max_2')))
            conv1_HRVI_max_2=tf.nn.conv2d(conv1_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer1_HRVI_mean_2'):
            weights=tf.get_variable("w1_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_mean_2/w1_HRVI_mean_2')))
            bias=tf.get_variable("b1_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_HRVI_mean_2/b1_HRVI_mean_2')))
            conv1_HRVI_mean_2=tf.nn.conv2d(conv1_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv1_HRVI_atten_map= tf.nn.sigmoid(conv1_HRVI_max_2+conv1_HRVI_mean_2)    
        conv1_HRVI_atten_out= conv1_HRVI_cat*conv1_HRVI_atten_map    
        
                
######################################################### 
#################### NDVI Layer 2 #######################
#########################################################  
##################  Multi_Scale 2 #######################     
        NDVI_dense_HR_t2=tf.concat([conv1_NDVI_atten_out,conv1_HRVI_atten_out],axis=-1)
        with tf.variable_scope('layer2_NDVI_3x3'):
            weights=tf.get_variable("w2_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_3x3/w2_NDVI_3x3')))
            bias=tf.get_variable("b2_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_3x3/b2_NDVI_3x3')))
            conv2_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_3x3 = lrelu(conv2_NDVI_3x3) 
        with tf.variable_scope('layer2_NDVI_5x5'):
            weights=tf.get_variable("w2_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_5x5/w2_NDVI_5x5')))
            bias=tf.get_variable("b2_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_5x5/b2_NDVI_5x5')))
            conv2_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_5x5 = lrelu(conv2_NDVI_5x5) 
        with tf.variable_scope('layer2_NDVI_7x7'):
            weights=tf.get_variable("w2_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_7x7/w2_NDVI_7x7')))
            bias=tf.get_variable("b2_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_7x7/b2_NDVI_7x7')))
            conv2_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_7x7 = lrelu(conv2_NDVI_7x7)
        conv2_NDVI_cat=tf.concat([conv2_NDVI_3x3,conv2_NDVI_5x5,conv2_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 2 ######################   
        CAttent_NDVI_2_max=tf.reduce_max(conv2_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_2_mean=tf.reduce_mean(conv2_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_NDVI_max_1'):
            weights=tf.get_variable("w2_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_max_1/w2_NDVI_max_1')))
            bias=tf.get_variable("b2_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_max_1/b2_NDVI_max_1')))
            conv2_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_max_1 = tf.nn.relu(conv2_NDVI_max_1)
        with tf.variable_scope('layer2_NDVI_mean_1'):
            weights=tf.get_variable("w2_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_mean_1/w2_NDVI_mean_1')))
            bias=tf.get_variable("b2_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_mean_1/b2_NDVI_mean_1')))
            conv2_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_NDVI_mean_1 = tf.nn.relu(conv2_NDVI_mean_1)            

        with tf.variable_scope('layer2_NDVI_max_2'):
            weights=tf.get_variable("w2_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_max_2/w2_NDVI_max_2')))
            bias=tf.get_variable("b2_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_max_2/b2_NDVI_max_2')))
            conv2_NDVI_max_2=tf.nn.conv2d(conv2_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_NDVI_mean_2'):
            weights=tf.get_variable("w2_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_mean_2/w2_NDVI_mean_2')))
            bias=tf.get_variable("b2_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_NDVI_mean_2/b2_NDVI_mean_2')))
            conv2_NDVI_mean_2=tf.nn.conv2d(conv2_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_NDVI_atten_map= tf.nn.sigmoid(conv2_NDVI_max_2+conv2_NDVI_mean_2)    
        conv2_NDVI_atten_out= conv2_NDVI_cat*conv2_NDVI_atten_map     
            

######################################################### 
#################### HRVI Layer 2 #######################
#########################################################  
##################  Multi_Scale 2 #######################     
        with tf.variable_scope('layer2_HRVI_3x3'):
            weights=tf.get_variable("w2_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_3x3/w2_HRVI_3x3')))
            bias=tf.get_variable("b2_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_3x3/b2_HRVI_3x3')))
            conv2_HRVI_3x3=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_3x3 = lrelu(conv2_HRVI_3x3) 
        with tf.variable_scope('layer2_HRVI_5x5'):
            weights=tf.get_variable("w2_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_5x5/w2_HRVI_5x5')))
            bias=tf.get_variable("b2_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_5x5/b2_HRVI_5x5')))
            conv2_HRVI_5x5=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_5x5 = lrelu(conv2_HRVI_5x5) 
        with tf.variable_scope('layer2_HRVI_7x7'):
            weights=tf.get_variable("w2_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_7x7/w2_HRVI_7x7')))
            bias=tf.get_variable("b2_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_7x7/b2_HRVI_7x7')))
            conv2_HRVI_7x7=tf.nn.conv2d(conv1_HRVI_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_7x7 = lrelu(conv2_HRVI_7x7)
        conv2_HRVI_cat=tf.concat([conv2_HRVI_3x3,conv2_HRVI_5x5,conv2_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 2 ######################   
        CAttent_HRVI_2_max=tf.reduce_max(conv2_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_2_mean=tf.reduce_mean(conv2_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer2_HRVI_max_1'):
            weights=tf.get_variable("w2_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_max_1/w2_HRVI_max_1')))
            bias=tf.get_variable("b2_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_max_1/b2_HRVI_max_1')))
            conv2_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_max_1 = tf.nn.relu(conv2_HRVI_max_1)
        with tf.variable_scope('layer2_HRVI_mean_1'):
            weights=tf.get_variable("w2_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_mean_1/w2_HRVI_mean_1')))
            bias=tf.get_variable("b2_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_mean_1/b2_HRVI_mean_1')))
            conv2_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_2_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_HRVI_mean_1 = tf.nn.relu(conv2_HRVI_mean_1)            

        with tf.variable_scope('layer2_HRVI_max_2'):
            weights=tf.get_variable("w2_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_max_2/w2_HRVI_max_2')))
            bias=tf.get_variable("b2_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_max_2/b2_HRVI_max_2')))
            conv2_HRVI_max_2=tf.nn.conv2d(conv2_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer2_HRVI_mean_2'):
            weights=tf.get_variable("w2_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_mean_2/w2_HRVI_mean_2')))
            bias=tf.get_variable("b2_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_HRVI_mean_2/b2_HRVI_mean_2')))
            conv2_HRVI_mean_2=tf.nn.conv2d(conv2_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv2_HRVI_atten_map= tf.nn.sigmoid(conv2_HRVI_max_2+conv2_HRVI_mean_2)    
        conv2_HRVI_atten_out= conv2_HRVI_cat*conv2_HRVI_atten_map 
            
######################################################### 
#################### NDVI Layer 3 #######################
#########################################################  
##################  Multi_Scale 3 ####################### 
        NDVI_dense_HR_t3=tf.concat([conv1_NDVI_atten_out,conv2_NDVI_atten_out,conv2_HRVI_atten_out],axis=-1)    
           
        with tf.variable_scope('layer3_NDVI_3x3'):
            weights=tf.get_variable("w3_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_3x3/w3_NDVI_3x3')))
            bias=tf.get_variable("b3_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_3x3/b3_NDVI_3x3')))
            conv3_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_3x3 = lrelu(conv3_NDVI_3x3) 
        with tf.variable_scope('layer3_NDVI_5x5'):
            weights=tf.get_variable("w3_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_5x5/w3_NDVI_5x5')))
            bias=tf.get_variable("b3_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_5x5/b3_NDVI_5x5')))
            conv3_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_5x5 = lrelu(conv3_NDVI_5x5) 
        with tf.variable_scope('layer3_NDVI_7x7'):
            weights=tf.get_variable("w3_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_7x7/w3_NDVI_7x7')))
            bias=tf.get_variable("b3_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_7x7/b3_NDVI_7x7')))
            conv3_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_7x7 = lrelu(conv3_NDVI_7x7)
        conv3_NDVI_cat=tf.concat([conv3_NDVI_3x3,conv3_NDVI_5x5,conv3_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_NDVI_3_max=tf.reduce_max(conv3_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_3_mean=tf.reduce_mean(conv3_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer3_NDVI_max_1'):
            weights=tf.get_variable("w3_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_max_1/w3_NDVI_max_1')))
            bias=tf.get_variable("b3_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_max_1/b3_NDVI_max_1')))
            conv3_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_max_1 = tf.nn.relu(conv3_NDVI_max_1)
        with tf.variable_scope('layer3_NDVI_mean_1'):
            weights=tf.get_variable("w3_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_mean_1/w3_NDVI_mean_1')))
            bias=tf.get_variable("b3_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_mean_1/b3_NDVI_mean_1')))
            conv3_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_3_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_NDVI_mean_1 = tf.nn.relu(conv3_NDVI_mean_1)            

        with tf.variable_scope('layer3_NDVI_max_2'):
            weights=tf.get_variable("w3_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_max_2/w3_NDVI_max_2')))
            bias=tf.get_variable("b3_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_max_2/b3_NDVI_max_2')))
            conv3_NDVI_max_2=tf.nn.conv2d(conv3_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer3_NDVI_mean_2'):
            weights=tf.get_variable("w3_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_mean_2/w3_NDVI_mean_2')))
            bias=tf.get_variable("b3_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_NDVI_mean_2/b3_NDVI_mean_2')))
            conv3_NDVI_mean_2=tf.nn.conv2d(conv3_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv3_NDVI_atten_map= tf.nn.sigmoid(conv3_NDVI_max_2+conv3_NDVI_mean_2)    
        conv3_NDVI_atten_out= conv3_NDVI_cat*conv3_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 3 #######################
#########################################################  
##################  Multi_Scale 3 #######################
        HRVI_dense_t3=tf.concat([conv1_HRVI_atten_out,conv2_HRVI_atten_out],axis=-1)              
        with tf.variable_scope('layer3_HRVI_3x3'):
            weights=tf.get_variable("w3_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_3x3/w3_HRVI_3x3')))
            bias=tf.get_variable("b3_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_3x3/b3_HRVI_3x3')))
            conv3_HRVI_3x3=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_3x3 = lrelu(conv3_HRVI_3x3) 
        with tf.variable_scope('layer3_HRVI_5x5'):
            weights=tf.get_variable("w3_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_5x5/w3_HRVI_5x5')))
            bias=tf.get_variable("b3_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_5x5/b3_HRVI_5x5')))
            conv3_HRVI_5x5=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_5x5 = lrelu(conv3_HRVI_5x5) 
        with tf.variable_scope('layer3_HRVI_7x7'):
            weights=tf.get_variable("w3_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_7x7/w3_HRVI_7x7')))
            bias=tf.get_variable("b3_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_7x7/b3_HRVI_7x7')))
            conv3_HRVI_7x7=tf.nn.conv2d(HRVI_dense_t3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_7x7 = lrelu(conv3_HRVI_7x7)
        conv3_HRVI_cat=tf.concat([conv3_HRVI_3x3,conv3_HRVI_5x5,conv3_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_HRVI_3_max=tf.reduce_max(conv3_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_3_mean=tf.reduce_mean(conv3_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer3_HRVI_max_1'):
            weights=tf.get_variable("w3_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_max_1/w3_HRVI_max_1')))
            bias=tf.get_variable("b3_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_max_1/b3_HRVI_max_1')))
            conv3_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_max_1 = tf.nn.relu(conv3_HRVI_max_1)
        with tf.variable_scope('layer3_HRVI_mean_1'):
            weights=tf.get_variable("w3_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_mean_1/w3_HRVI_mean_1')))
            bias=tf.get_variable("b3_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_mean_1/b3_HRVI_mean_1')))
            conv3_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_3_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_HRVI_mean_1 = tf.nn.relu(conv3_HRVI_mean_1)            

        with tf.variable_scope('layer3_HRVI_max_2'):
            weights=tf.get_variable("w3_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_max_2/w3_HRVI_max_2')))
            bias=tf.get_variable("b3_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_max_2/b3_HRVI_max_2')))
            conv3_HRVI_max_2=tf.nn.conv2d(conv3_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer3_HRVI_mean_2'):
            weights=tf.get_variable("w3_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_mean_2/w3_HRVI_mean_2')))
            bias=tf.get_variable("b3_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_HRVI_mean_2/b3_HRVI_mean_2')))
            conv3_HRVI_mean_2=tf.nn.conv2d(conv3_HRVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv3_HRVI_atten_map= tf.nn.sigmoid(conv3_HRVI_max_2+conv3_HRVI_mean_2)    
        conv3_HRVI_atten_out= conv3_HRVI_cat*conv3_HRVI_atten_map    


######################################################### 
#################### NDVI Layer 4 #######################
#########################################################  
##################  Multi_Scale 4 ####################### 
        NDVI_dense_HR_t4=tf.concat([conv1_NDVI_atten_out,conv2_NDVI_atten_out,conv3_NDVI_atten_out,conv3_HRVI_atten_out],axis=-1)         
        with tf.variable_scope('layer4_NDVI_3x3'):
            weights=tf.get_variable("w4_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_3x3/w4_NDVI_3x3')))
            bias=tf.get_variable("b4_NDVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_3x3/b4_NDVI_3x3')))
            conv4_NDVI_3x3=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_3x3 = lrelu(conv4_NDVI_3x3) 
        with tf.variable_scope('layer4_NDVI_5x5'):
            weights=tf.get_variable("w4_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_5x5/w4_NDVI_5x5')))
            bias=tf.get_variable("b4_NDVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_5x5/b4_NDVI_5x5')))
            conv4_NDVI_5x5=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_5x5 = lrelu(conv4_NDVI_5x5) 
        with tf.variable_scope('layer4_NDVI_7x7'):
            weights=tf.get_variable("w4_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_7x7/w4_NDVI_7x7')))
            bias=tf.get_variable("b4_NDVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_7x7/b4_NDVI_7x7')))
            conv4_NDVI_7x7=tf.nn.conv2d(NDVI_dense_HR_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_7x7 = lrelu(conv4_NDVI_7x7)
        conv4_NDVI_cat=tf.concat([conv4_NDVI_3x3,conv4_NDVI_5x5,conv4_NDVI_7x7],axis=-1)                                         
#################  Channel Attention 4 ######################   
        CAttent_NDVI_4_max=tf.reduce_max(conv4_NDVI_cat, axis=(1, 2), keepdims=True)
        CAttent_NDVI_4_mean=tf.reduce_mean(conv4_NDVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer4_NDVI_max_1'):
            weights=tf.get_variable("w4_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_max_1/w4_NDVI_max_1')))
            bias=tf.get_variable("b4_NDVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_max_1/b4_NDVI_max_1')))
            conv4_NDVI_max_1=tf.nn.conv2d(CAttent_NDVI_4_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_max_1 = tf.nn.relu(conv4_NDVI_max_1)
        with tf.variable_scope('layer4_NDVI_mean_1'):
            weights=tf.get_variable("w4_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_mean_1/w4_NDVI_mean_1')))
            bias=tf.get_variable("b4_NDVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_mean_1/b4_NDVI_mean_1')))
            conv4_NDVI_mean_1=tf.nn.conv2d(CAttent_NDVI_4_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_NDVI_mean_1 = tf.nn.relu(conv4_NDVI_mean_1)            

        with tf.variable_scope('layer4_NDVI_max_2'):
            weights=tf.get_variable("w4_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_max_2/w4_NDVI_max_2')))
            bias=tf.get_variable("b4_NDVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_max_2/b4_NDVI_max_2')))
            conv4_NDVI_max_2=tf.nn.conv2d(conv4_NDVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer4_NDVI_mean_2'):
            weights=tf.get_variable("w4_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_mean_2/w4_NDVI_mean_2')))
            bias=tf.get_variable("b4_NDVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_NDVI_mean_2/b4_NDVI_mean_2')))
            conv4_NDVI_mean_2=tf.nn.conv2d(conv4_NDVI_mean_1, weights, strides=[1,1,1,1], padding='SAME') + bias

        conv4_NDVI_atten_map= tf.nn.sigmoid(conv4_NDVI_max_2+conv4_NDVI_mean_2)    
        conv4_NDVI_atten_out= conv4_NDVI_cat*conv4_NDVI_atten_map    


######################################################### 
#################### HRVI Layer 4 #######################
#########################################################  
##################  Multi_Scale 4 #######################
        HRVI_dense_t4=tf.concat([conv1_HRVI_atten_out,conv2_HRVI_atten_out,conv3_HRVI_atten_out],axis=-1)              
        with tf.variable_scope('layer4_HRVI_3x3'):
            weights=tf.get_variable("w4_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_3x3/w4_HRVI_3x3')))
            bias=tf.get_variable("b4_HRVI_3x3",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_3x3/b4_HRVI_3x3')))
            conv4_HRVI_3x3=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_3x3 = lrelu(conv4_HRVI_3x3) 
        with tf.variable_scope('layer4_HRVI_5x5'):
            weights=tf.get_variable("w4_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_5x5/w4_HRVI_5x5')))
            bias=tf.get_variable("b4_HRVI_5x5",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_5x5/b4_HRVI_5x5')))
            conv4_HRVI_5x5=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_5x5 = lrelu(conv4_HRVI_5x5) 
        with tf.variable_scope('layer4_HRVI_7x7'):
            weights=tf.get_variable("w4_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_7x7/w4_HRVI_7x7')))
            bias=tf.get_variable("b4_HRVI_7x7",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_7x7/b4_HRVI_7x7')))
            conv4_HRVI_7x7=tf.nn.conv2d(HRVI_dense_t4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_7x7 = lrelu(conv4_HRVI_7x7)
        conv4_HRVI_cat=tf.concat([conv4_HRVI_3x3,conv4_HRVI_5x5,conv4_HRVI_7x7],axis=-1)                                         
#################  Channel Attention 3 ######################   
        CAttent_HRVI_4_max=tf.reduce_max(conv4_HRVI_cat, axis=(1, 2), keepdims=True)
        CAttent_HRVI_4_mean=tf.reduce_mean(conv4_HRVI_cat, axis=(1, 2), keepdims=True)
        
        with tf.variable_scope('layer4_HRVI_max_1'):
            weights=tf.get_variable("w4_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_max_1/w4_HRVI_max_1')))
            bias=tf.get_variable("b4_HRVI_max_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_max_1/b4_HRVI_max_1')))
            conv4_HRVI_max_1=tf.nn.conv2d(CAttent_HRVI_4_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_max_1 = tf.nn.relu(conv4_HRVI_max_1)
        with tf.variable_scope('layer4_HRVI_mean_1'):
            weights=tf.get_variable("w4_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_mean_1/w4_HRVI_mean_1')))
            bias=tf.get_variable("b4_HRVI_mean_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_mean_1/b4_HRVI_mean_1')))
            conv4_HRVI_mean_1=tf.nn.conv2d(CAttent_HRVI_4_mean, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_HRVI_mean_1 = tf.nn.relu(conv4_HRVI_mean_1)            

        with tf.variable_scope('layer4_HRVI_max_2'):
            weights=tf.get_variable("w4_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_max_2/w4_HRVI_max_2')))
            bias=tf.get_variable("b4_HRVI_max_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_max_2/b4_HRVI_max_2')))
            conv4_HRVI_max_2=tf.nn.conv2d(conv4_HRVI_max_1, weights, strides=[1,1,1,1], padding='SAME') + bias
        with tf.variable_scope('layer4_HRVI_mean_2'):
            weights=tf.get_variable("w4_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_mean_2/w4_HRVI_mean_2')))
            bias=tf.get_variable("b4_HRVI_mean_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_HRVI_mean_2/b4_HRVI_mean_2')))
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
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5=tf.nn.conv2d(NDVI_HRVI_cat, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5 = lrelu(conv5)                                         
#################  Spatial Attention 4 ####################   
        SAttent_5_max=tf.reduce_max(conv5, axis=3, keepdims=True)
        SAttent_5_mean=tf.reduce_mean(conv5, axis=3, keepdims=True)
        SAttent_5_cat_mean_max=tf.concat([SAttent_5_max,SAttent_5_mean],axis=-1)        
        with tf.variable_scope('layer5_atten_map'):
            weights=tf.get_variable("w5_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_atten_map/w5_atten_map')))
            bias=tf.get_variable("b5_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_atten_map/b5_atten_map')))
            conv5_atten_map=tf.nn.conv2d(SAttent_5_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_atten_map = tf.nn.sigmoid(conv5_atten_map)  
        conv5_atten_out= conv5*conv5_atten_map    

######################################################### 
####################  Layer 6 ###########################
#########################################################  
####################  Conv 6  ###########################            
        with tf.variable_scope('layer6'):
            weights=tf.get_variable("w6",initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6')))
            bias=tf.get_variable("b6",initializer=tf.constant(reader.get_tensor('fusion_model/layer6/b6')))
            conv6=tf.nn.conv2d(conv5_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6 = lrelu(conv6)                                         
#################  Spatial Attention 6 ####################   
        SAttent_6_max=tf.reduce_max(conv6, axis=3, keepdims=True)
        SAttent_6_mean=tf.reduce_mean(conv6, axis=3, keepdims=True)
        SAttent_6_cat_mean_max=tf.concat([SAttent_6_max,SAttent_6_mean],axis=-1)
        
        with tf.variable_scope('layer6_atten_map'):
            weights=tf.get_variable("w6_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_atten_map/w6_atten_map')))
            bias=tf.get_variable("b6_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_atten_map/b6_atten_map')))
            conv6_atten_map=tf.nn.conv2d(SAttent_6_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_atten_map = tf.nn.sigmoid(conv6_atten_map)  
        conv6_atten_out= conv6*conv6_atten_map    

######################################################### 
####################  Layer 7 ###########################
#########################################################  
        sa_dense_t7=tf.concat([conv5_atten_out,conv6_atten_out],axis=-1)
####################  Conv 6  ###########################            
        with tf.variable_scope('layer7'):
            weights=tf.get_variable("w7",initializer=tf.constant(reader.get_tensor('fusion_model/layer7/w7')))
            bias=tf.get_variable("b7",initializer=tf.constant(reader.get_tensor('fusion_model/layer7/b7')))
            conv7=tf.nn.conv2d(sa_dense_t7, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7 = lrelu(conv7)                                         
#################  Spatial Attention 6 ####################   
        SAttent_7_max=tf.reduce_max(conv7, axis=3, keepdims=True)
        SAttent_7_mean=tf.reduce_mean(conv7, axis=3, keepdims=True)
        SAttent_7_cat_mean_max=tf.concat([SAttent_7_max,SAttent_7_mean],axis=-1)
        
        with tf.variable_scope('layer7_atten_map'):
            weights=tf.get_variable("w7_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_atten_map/w7_atten_map')))
            bias=tf.get_variable("b7_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_atten_map/b7_atten_map')))
            conv7_atten_map=tf.nn.conv2d(SAttent_7_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_atten_map = tf.nn.sigmoid(conv7_atten_map)  
        conv7_atten_out= conv7*conv7_atten_map  

######################################################### 
####################  Layer 8 ###########################
#########################################################  
        sa_dense_t8=tf.concat([conv5_atten_out,conv6_atten_out,conv7_atten_out],axis=-1)
####################  Conv 8  ###########################            
        with tf.variable_scope('layer8'):
            weights=tf.get_variable("w8",initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8')))
            bias=tf.get_variable("b8",initializer=tf.constant(reader.get_tensor('fusion_model/layer8/b8')))
            conv8=tf.nn.conv2d(sa_dense_t8, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8 = lrelu(conv8)                                         
#################  Spatial Attention 8 ####################   
        SAttent_8_max=tf.reduce_max(conv8, axis=3, keepdims=True)
        SAttent_8_mean=tf.reduce_mean(conv8, axis=3, keepdims=True)
        SAttent_8_cat_mean_max=tf.concat([SAttent_8_max,SAttent_8_mean],axis=-1)
        
        with tf.variable_scope('layer8_atten_map'):
            weights=tf.get_variable("w8_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_atten_map/w8_atten_map')))
            bias=tf.get_variable("b8_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_atten_map/b8_atten_map')))
            conv8_atten_map=tf.nn.conv2d(SAttent_8_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_atten_map = tf.nn.sigmoid(conv8_atten_map)  
        conv8_atten_out= conv8*conv8_atten_map  
            
################ Reconstruction ######################### 
######################################################### 
####################  Layer 9 ###########################
######################################################### 
        with tf.variable_scope('layer9'):
            weights=tf.get_variable("w9",initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9')))
            bias=tf.get_variable("b9",initializer=tf.constant(reader.get_tensor('fusion_model/layer9/b9')))
            conv9=tf.nn.conv2d(conv8_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9 =lrelu(conv9)  
######################################################### 
####################  Layer 8 ###########################
######################################################### 
        with tf.variable_scope('layer10'):
            weights=tf.get_variable("w10",initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10')))
            bias=tf.get_variable("b10",initializer=tf.constant(reader.get_tensor('fusion_model/layer10/b10')))
            conv10=tf.nn.conv2d(conv9, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv10 =tf.nn.tanh(conv10)  
    return conv10
    

def input_setup(index):
    padding=0
    sub_NDVI_sequence = []
    sub_HRVI_sequence = []
    
    input_NDVI=(imread(data_NDVI[index])-127.5)/127.5
    input_NDVI=np.lib.pad(input_NDVI,((padding,padding),(padding,padding)),'edge')
    w,h=input_NDVI.shape
    input_NDVI=input_NDVI.reshape([w,h,1])
    
    input_HRVI=(imread(data_HRVI[index])-127.5)/127.5
    input_HRVI=np.lib.pad(input_HRVI,((padding,padding),(padding,padding)),'edge')
    w,h=input_HRVI.shape
    input_HRVI=input_HRVI.reshape([w,h,1])
    
    sub_NDVI_sequence.append(input_NDVI)
    sub_HRVI_sequence.append(input_HRVI)
    train_data_NDVI= np.asarray(sub_NDVI_sequence)
    train_data_HRVI= np.asarray(sub_HRVI_sequence)
    return train_data_NDVI,train_data_HRVI

satellite='GF2'
#satellite='QB' 
reader = tf.train.NewCheckpointReader('./checkpoint/CGAN/CGAN.model-'+satellite)
  
with tf.name_scope('NDVI_input'):
    images_NDVI = tf.placeholder(tf.float32, [1,None,None,None], name='images_NDVI')
with tf.name_scope('HRVI_input'):
    images_HRVI= tf.placeholder(tf.float32, [1,None,None,None], name='images_HRVI')          
with tf.name_scope('input'):
    input_image_NDVI =images_NDVI
    input_image_HRVI =images_HRVI
  
with tf.name_scope('fusion'):
    fusion_image=fusion_model(input_image_NDVI,input_image_HRVI)
    
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    data_NDVI=prepare_data('Test_NDVI')
    data_HRVI=prepare_data('Test_HRVI')
    for i in range(len(data_NDVI)):
      start=time.time()
      train_data_NDVI,train_data_HRVI=input_setup(i)
      result =sess.run(fusion_image,feed_dict={images_NDVI: train_data_NDVI,images_HRVI: train_data_HRVI})
      result=result*127.5+127.5
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), 'result_'+satellite)
      if not os.path.exists(image_path):
          os.makedirs(image_path)
      image_path = os.path.join(image_path, str(i+1)+".tif")
      end=time.time()
      imsave(result, image_path)
      print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
tf.reset_default_graph()
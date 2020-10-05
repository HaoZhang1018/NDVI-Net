# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    return data

def preprocess(path, scale=3):
  image = imread(path, is_grayscale=True)
  image = (image-127.5 )/ 127.5 
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
  return input_

def prepare_data(sess, dataset):
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
  return data

def make_data(sess, data, data_dir):
  if FLAGS.is_train:
    savepath = os.path.join('.', os.path.join('checkpoint',data_dir,'train.h5'))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint',data_dir))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint',data_dir)))
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)


def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup_NDVI(sess,config,data_dir,index=0):
  if config.is_train:
    data = prepare_data(sess, dataset=data_dir)

  sub_input_sequence = []
  padding = 0 
  if config.is_train:
    for i in xrange(len(data)):
      input_=(imread(data[i])-127.5)/127.5

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      for x in range(0, h-config.image_size_NDVI+1, config.stride_NDVI):
        for y in range(0, w-config.image_size_NDVI+1, config.stride_NDVI):
          sub_input = input_[x:x+config.image_size_NDVI, y:y+config.image_size_NDVI] # [33 x 33]       
          # Make channel value
          if data_dir == "Train":
            sub_input=cv2.resize(sub_input, (config.image_size_NDVI/4,config.image_size_NDVI/4),interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size_NDVI/4, config.image_size_NDVI/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size_NDVI, config.image_size_NDVI, 1])            
          sub_input_sequence.append(sub_input)
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  #print(arrdata.shape)
  make_data(sess, arrdata, data_dir)

def input_setup_HRVI(sess,config,data_dir,index=0):
  if config.is_train:
    data = prepare_data(sess, dataset=data_dir)

  sub_input_sequence = []
  padding = 0 
  if config.is_train:
    for i in xrange(len(data)):
      input_=(imread(data[i])-127.5)/127.5

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      for x in range(0, h-config.image_size_HRVI+1, config.stride_HRVI):
        for y in range(0, w-config.image_size_HRVI+1, config.stride_HRVI):
          sub_input = input_[x:x+config.image_size_HRVI, y:y+config.image_size_HRVI] # [33 x 33]       
          # Make channel value
          if data_dir == "Train":
            sub_input=cv2.resize(sub_input, (config.image_size_HRVI/4,config.image_size_HRVI/4),interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size_HRVI/4, config.image_size_HRVI/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size_HRVI, config.image_size_HRVI, 1])            
          sub_input_sequence.append(sub_input)
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  #print(arrdata.shape)
  make_data(sess, arrdata, data_dir)


  if not config.is_train:
    print(nx,ny)
    print(h_real,w_real)
    return nx, ny,h_real,w_real
    
def imsave(image, path):
  return scipy.misc.imsave(path, image.astype(np.uint8))

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return (img*127.5+127.5)
  
def sobel_gradient(input):
    filter_x=tf.reshape(tf.constant([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]),[3,3,1,1])
    filter_y=tf.reshape(tf.constant([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]),[3,3,1,1])
    d_x=tf.nn.conv2d(input,filter_x,strides=[1,1,1,1], padding='SAME')
    d_y=tf.nn.conv2d(input,filter_y,strides=[1,1,1,1], padding='SAME')
    return d_x, d_y
    
def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1
        
        u_hat, v_hat,_ = power_iteration(u,iteration)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        
        w_mat = w_mat/sigma
        
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm
    
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)
    
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm

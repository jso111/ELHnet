# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:14:03 2016

@author: George

Fast ConvNet methods adapted from CS231n 2016 code
Avoids using Cython

-Original source: fast_layers.py from 
https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/fast_layers.py
-Can also download fast_layers.py and its dependencies in zip folder from
http://cs231n.github.io/assignments2016/assignment2/ 

Dependencies: im2col

Last updated: 8/26/2016
"""

import numpy as np
from scipy.ndimage.filters import convolve # needed for naive forward pass implementation
import im2col # https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col.py

def conv_forward_strides(x, w, b, conv_param):
  """
  A fast implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  # Check dimensions
  assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
  assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

  # Pad the input
  p = pad
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  
  # Figure out output dimensions
  H += 2 * pad
  W += 2 * pad
  out_h = (H - HH) / stride + 1
  out_w = (W - WW) / stride + 1

  # Perform an im2col operation by picking clever strides
  shape = (C, HH, WW, N, out_h, out_w)
  strides = (H * W, W, 1, C * H * W, stride * W, stride)
  strides = x.itemsize * np.array(strides)
  x_stride = np.lib.stride_tricks.as_strided(x_padded,
                shape=shape, strides=strides) # cause of 'return array(a, dtype, copy=False, order=order)' warning
  x_cols = np.ascontiguousarray(x_stride)
  x_cols.shape = (int(C * HH * WW), int(N * out_h * out_w))

  # Now all our convolutions are a big matrix multiply
  res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

  # Reshape the output
  res.shape = (F, N, int(out_h), int(out_w))
  out = res.transpose(1, 0, 2, 3)

  # Be nice and return a contiguous array
  # The old version of conv_forward_fast doesn't do this, so for a fair
  # comparison we won't either
  out = np.ascontiguousarray(out)

  cache = (x, w, b, conv_param, x_cols)
  return out, cache
  
  
conv_forward_fast = conv_forward_strides
  
  
def max_pool_forward_fast(x, pool_param):
  """
  A fast implementation of the forward pass for a max pooling layer.
  This chooses between the reshape method and the im2col method. If the pooling
  regions are square and tile the input image, then we can use the reshape
  method which is very fast. Otherwise we fall back on the im2col method, which
  is not much faster than the naive method.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  same_size = pool_height == pool_width == stride
  tiles = H % pool_height == 0 and W % pool_width == 0
  if same_size and tiles:
    out, reshape_cache = max_pool_forward_reshape(x, pool_param)
    cache = ('reshape', reshape_cache)
  else:
    out, im2col_cache = max_pool_forward_im2col(x, pool_param)
    cache = ('im2col', im2col_cache)
  return out, cache


def max_pool_forward_reshape(x, pool_param):
  """
  A fast implementation of the forward pass for the max pooling layer that uses
  some clever reshaping.

  This can only be used for square pooling regions that tile the input.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  assert pool_height == pool_width == stride, 'Invalid pool params'
  assert H % pool_height == 0
  assert W % pool_height == 0
  x_reshaped = x.reshape(N, C, int(H / pool_height), pool_height,
                         int(W / pool_width), pool_width)
  out = x_reshaped.max(axis=3).max(axis=4)

  cache = (x, x_reshaped, out)
  return out, cache
  
  
def max_pool_forward_im2col(x, pool_param):
  """
  An implementation of the forward pass for max pooling based on im2col.
  This isn't much faster than the naive version, so it should be avoided if
  possible.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  assert (H - pool_height) % stride == 0, 'Invalid height'
  assert (W - pool_width) % stride == 0, 'Invalid width'

  out_height = (H - pool_height) / stride + 1
  out_width = (W - pool_width) / stride + 1

  x_split = x.reshape(N * C, 1, H, W)
  x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
  x_cols_argmax = np.argmax(x_cols, axis=0)
  x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
  out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

  cache = (x, x_cols, x_cols_argmax, pool_param)
  return out, cache


#%% Naive convolutional neural network methods (SLOW)
def conv_forward_naiveReLU(X, weight, bias):
    """
    Inputs:
    -X: (1, depth, width, height)
    -weight: (#filters, input depth, input width, input height)
    -bias: (#filters)
    
    Returns numpy.ndarray of float64:
    -activation: (1, output depth=#filters, output width=input width, output height=input height)
    """
    nfilters = np.shape(weight)[0] # 32 (comments for first convolutional layer var shapes)
    X = np.tile(X,(nfilters,1,1,1)) # 32x1x64x64
    out = convolve(X, weight, mode='constant', cval=0.0) # 32x1x64x64 <---takes 105 s in numpy for second conv layer!!  
    out = out + np.reshape(bias, (nfilters,1,1,1)) # 32x1x64x64
    out = np.ndarray.transpose(out, (1,0,2,3)) # 1x32x64x64
    activation = np.maximum(out, 0) # element-wise ReLU
    
    return activation

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:21:22 2016

@author: George

Run ELHnet, our convolutional neural network to classify OCT images with 
endolymphatic hydrops.

-> First run "%matplotlib qt" command in IPython console to enable point selection.
This also resolves the error, "matplotlib is currently using a non-GUI backend"

Use the predict(X) method to classify images (see code)

Last edit: 9/16/2017

Dependencies: fast_layersGL.py, mat.py

"""
from __future__ import print_function
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit
from datetime import datetime

from fast_layersGL import conv_forward_fast, max_pool_forward_fast
from fast_layersGL import conv_forward_naiveReLU # for debugging forward pass
import mat

def rotweights180(w):
    """
    Rotate array in last two dimensions of w by 180 degrees
    -w is (F,C,H,W) array, where H=W and a(i,j,:,:) needs to be rotated for each i,j
    """
    w_tp = w.transpose(2,3,0,1) # (H,W,F,C)
    w_tp = np.rot90(w_tp, k=2) # rotates first two dimensions of array
    w_rot180 = w_tp.transpose(2,3,0,1) # (F,C,H,W)
    
    return w_rot180
    
    
# load weights
weight_path = 'octcnn_weights_valacc0.99817232376.h5'
hf = h5py.File(weight_path,'r')
convolution2d_1_W = hf['convolution2d_1/convolution2d_1_W'][()] # 32x1x3x3 (#filters, input depth, input width, input height)
convolution2d_1_b = hf['convolution2d_1/convolution2d_1_b'][()]
convolution2d_2_W = hf['convolution2d_2/convolution2d_2_W'][()] # 32x32x3x3
convolution2d_2_b = hf['convolution2d_2/convolution2d_2_b'][()]
convolution2d_3_W = hf['convolution2d_3/convolution2d_3_W'][()] # 32x32x3x3
convolution2d_3_b = hf['convolution2d_3/convolution2d_3_b'][()]
convolution2d_4_W = hf['convolution2d_4/convolution2d_4_W'][()] # 64x32x3x3
convolution2d_4_b = hf['convolution2d_4/convolution2d_4_b'][()]
convolution2d_5_W = hf['convolution2d_5/convolution2d_5_W'][()] # 64x64x3x3
convolution2d_5_b = hf['convolution2d_5/convolution2d_5_b'][()]
convolution2d_6_W = hf['convolution2d_6/convolution2d_6_W'][()] # 32x64x3x3
convolution2d_6_b = hf['convolution2d_6/convolution2d_6_b'][()]
dense_1_W = hf['dense_1/dense_1_W'][()] # 8192x512
dense_1_b = hf['dense_1/dense_1_b'][()] # 512,
dense_2_W = hf['dense_2/dense_2_W'][()] # 512x1
dense_2_b = hf['dense_2/dense_2_b'][()] # 1,
    
# Rotate Keras weight kernels by 180 degrees to make compatible with conv_forward_fast.
# This is because conv_forward_fast takes weight kernels that are rotated 180 
# degrees relative to the format used in Keras models and the naive forward pass implementation.
convolution2d_1_W = rotweights180(convolution2d_1_W) # 32x1x3x3 (#filters, input depth, input width, input height)
convolution2d_2_W = rotweights180(convolution2d_2_W) # 32x32x3x3
convolution2d_3_W = rotweights180(convolution2d_3_W) # 32x32x3x3
convolution2d_4_W = rotweights180(convolution2d_4_W) # 64x32x3x3
convolution2d_5_W = rotweights180(convolution2d_5_W) # 64x64x3x3
convolution2d_6_W = rotweights180(convolution2d_6_W) # 32x64x3x3

print('Weights loaded to ELHnet model.')


def classify(X):
    """
    Classify endolymphatic hydrops in 64x64 images. 
    Implements ELHnet ConvNet forward pass to perform binary classification
    using model weights that are read from HDF5 file on disk.
    Input:
    - X: Input data, of shape (N, C=1, H=64, W=64)
            N: number of images
            C: number of channels. Must be 1 b/c model accepts only grayscale.
            H: number of rows. Must be 64.
            W: number of columns. Must be 64.
    
    Returns a numpy.float64 of:
    - score: Predicted probability of endolymphatic hydrops
    """
    conv_params = {'stride': 1, 'pad':1}
    pool_params = {'pool_height':2, 'pool_width':2, 'stride':2} # fast computation of maxpool if all values are the same
    
    # Input layer
    if len(np.shape(X)) < 4:
        print('Warning: input X to classify function in elhnet.py is shape', np.shape(X), 'but should be shape (N, C=1, H=64, W=64). Reshaping....')
        X = np.reshape(X, (1,1,64,64)) # convert 64x64 to 1x1x64x64 ndarray
        print('New shape is', np.shape(X))
    input_layer = X
    
    # Convolution layer 1
    out1, cach1 = conv_forward_fast(input_layer, convolution2d_1_W, convolution2d_1_b, conv_params) # Nx32x64x64
    out1 = np.maximum(out1, 0) # element-wise ReLU
    
    # Convolution layer 2
    out2, cach2 = conv_forward_fast(out1, convolution2d_2_W, convolution2d_2_b, conv_params) # Nx32x64x64
    out2 = np.maximum(out2, 0) # element-wise ReLU
    
    # Max pool
    pool2, cachpool2 = max_pool_forward_fast(out2, pool_params)
    
    # Convolution layer 3
    out3, cach3 = conv_forward_fast(pool2, convolution2d_3_W, convolution2d_3_b, conv_params) # Nx32x32x32
    out3 = np.maximum(out3, 0) # element-wise ReLU
    
    # Convolution layer 4
    out4, cach4 = conv_forward_fast(out3, convolution2d_4_W, convolution2d_4_b, conv_params) # Nx64x32x32
    out4 = np.maximum(out4, 0) # element-wise ReLU
    
    # Convolution layer 5
    out5, cach5 = conv_forward_fast(out4, convolution2d_5_W, convolution2d_5_b, conv_params) # Nx64x32x32
    out5 = np.maximum(out5, 0) # element-wise ReLU
    
    # Max pool
    pool5, cachpool5 = max_pool_forward_fast(out5, pool_params)
    
    # Convolution layer 6
    out6, cach6 = conv_forward_fast(pool5, convolution2d_6_W, convolution2d_6_b, conv_params) # Nx32x16x16
    out6 = np.maximum(out6, 0) # element-wise ReLU
    
    # Flatten layer
    flat6 = np.reshape(out6, (out6.shape[0], np.product(out6.shape[1:]))) # Nx8192
    
    # Dense layer 7
    out7 = np.tanh(np.dot(flat6, dense_1_W) + dense_1_b) # Nx512
    
    # Dense layer 8
    out8 = expit(np.dot(out7, dense_2_W) + dense_2_b) # Nx1
    
    # Output binary classification score(s)
    score = out8.flatten() # probability of endolymphatic hydrops
    
#    # Verbose for debugging
#    print('out1:', np.shape(out1))
#    print(type(out1))
#    print('out1[0,0,:,:]:', out1[0,0,:,:])
#    print('out1[0,-1,:,:]:', out1[0,-1,:,:])
#    out1naive = conv_forward_naiveReLU(X, convolution2d_1_W, convolution2d_1_b)
#    print('out1naive:', np.shape(out1naive))
#    print(type(out1naive))
#    print('out1naive[0,0,:,:]:', out1naive[0,0,:,:])
#    print('out1naive[0,0,:,:]>0', out1naive[0,0,:,:]>0)
#    print('out2:', np.shape(out2))
#    print('pool2:', np.shape(pool2))
#    print('out3:', np.shape(out3))
#    print('out4:', np.shape(out4))
#    print('out5:', np.shape(out5))
#    print('pool5:', np.shape(pool5))
#    print('out6:', np.shape(out6))
#    print('flat6:', np.shape(flat6))
#    print('out7:', np.shape(out7))
    
    return score
    
    
def classify_batch(X, batch_size=32):
    """
    Run elhnet ConvNet forward pass on 64x64 images processed by batch to avoid 
    overloading memory when running on a large number of images.
    Input:
    - X: Input data, of shape (N, C=1, H=64, W=64)
    - batch_size: size of batches to feed to model
    
    Returns a numpy.float64 of:
    - scores: Predicted probabilities of endolymphatic hydrops
    """
    N_total = X.shape[0]
    n_batches = int(np.ceil(N_total/batch_size))
    scores = []
    for i in range(n_batches):
        # print progress
        if i%10 == 0:
            print('  Working on batch', i, 'out of', n_batches, '....')
        
        # obtain i-th batch
        batch_index_first = i*batch_size
        batch_index_last = (i+1)*batch_size
        if batch_index_last > N_total:
            batch_index_last = N_total
        batch = X[batch_index_first:batch_index_last]
        
        # run forward pass on batch
        batch_scores = classify(batch)
        
        # accumulate predictions
        scores = np.append(scores, batch_scores)
        
    print('  Done.')
    
    return scores


def detect(X):
    """
    Run elhnet ConvNet forward pass on 64x64 images processed by batch to avoid 
    overloading memory when running on a large number of images.
    Input:
    - X: Input data, of shape (N, C=1, H, W)
            N: number of images
            C: number of channels. Must be 1 b/c model accepts only grayscale.
            H: number of rows.
            W: number of columns.
    
    Returns a numpy.float64 of:
    - scores: Predicted probabilities of endolymphatic hydrops
    """
    N = X.shape[0]
    H = X.shape[-2]
    W = X.shape[-1]
    
    # Ensure input array is proper format
    if len(np.shape(X)) < 4:
        print('Warning: input X to classify function in elhnet.py is shape', np.shape(X), 'but should be shape (N, C=1, H=64, W=64). Reshaping....')
        X = np.reshape(X, (1,1,H,W)) # convert HxW to 1x1xHxW ndarray
        print('New shape is', np.shape(X))
    
    # Zero-pad image dimensions that are less than 64
    print('Initial (H,W)', (H,W))
    if H < 64 or W < 64:
        difH = np.maximum(64 - H, 0)
        difW = np.maximum(64 - W, 0)
        padH_bef = int(np.floor(difH/2))
        padH_aft = int(np.ceil(difH/2))
        difW_bef = int(np.floor(difW/2))
        difW_aft = int(np.ceil(difW/2))
        X = np.pad(X, ((0, 0), (0, 0), (padH_bef, padH_aft), (difW_bef, difW_aft)), mode='constant')
        H = X.shape[2]
        W = X.shape[3]
        print('Reshaped (H,W):', np.shape(X))
    
    # Calculate positions of 64x64 tiles to cover image
    numH = int(np.ceil(H/64))
    numW = int(np.ceil(W/64))
    lastH = H-64
    lastW = W-64
    posH = np.linspace(0, lastH, numH, dtype=int)
    posW = np.linspace(0, lastW, numW, dtype=int)
    
    print('posH', posH)
    print('posW', posW)
    
    # Classify each tile in image
    tile_scores = np.empty((N,numH,numW), float)
    for i in range(numH):
        for j in range(numW):
            thisH = posH[i]
            thisW = posW[j]
            this_tile = X[:, :, thisH:thisH+64, thisW:thisW+64]
            this_tile_scores = classify(this_tile)
            tile_scores[:,i,j] = this_tile_scores
            
            # display tile on original image for N=0
            im = X[0,0,:,:]
            
            # Create figure and axes
            fig,ax = plt.subplots(1)
            
            # Display the image
            ax.imshow(im, cmap='gray')
            
            # Create a Rectangle patch
            if this_tile_scores>0.5:
                col = 'r'
            else:
                col = 'g'
            rect = patches.Rectangle((thisW,thisH),64,64,linewidth=2,edgecolor=col,facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)
            
            plt.title('(H,W): ' + str((thisH, thisW)) + '  Score: ' + str(this_tile_scores))
            plt.show()
            
    fig2,ax2 = plt.subplots(1)
    ax2.imshow(im, cmap='gray')
    for i in range(numH):
        for j in range(numW):
            thisH = posH[i]
            thisW = posW[j]
            this_tile = X[:, :, thisH:thisH+64, thisW:thisW+64]
            this_tile_scores = classify(this_tile)
            
            # Create a Rectangle patch
            if this_tile_scores>0.5:
                col = 'r'
            else:
                col = 'g'
            rect = patches.Rectangle((thisW,thisH),64,64,linewidth=2,edgecolor=col,facecolor='none')
            
            # Add the patch to the Axes
            ax2.add_patch(rect)
    plt.show()
    
    # average tile scores to obtain final image scores
    scores = np.mean(tile_scores, axis=(1,2))
    
    return scores
    
    
def preprocess(X):
    # Perform mean subtraction and normalization using per image statistics to turn pixel values into z-scores
    X = X - np.mean(X)
    X /= np.std(X)  
    
    return X
    

if __name__ == '__main__':
    # ask user to select an image
    filename = mat.uigetfile()
    
    # read image
    X = plt.imread(filename)
    if len(np.shape(X))==3:
        X = X[:,:,0] # convert 64x64x4 augmented image to 64x64 2-D array
    
    # Perform mean subtraction and normalization using per image statistics to turn pixel values into z-scores
    X = preprocess(X)
   
    ### Run model on user-selected ROI from input image, display, and save results
    # FIRST run %matplotlib qt
    coords = mat.pickpoint(X) # pick center of ROI
    x,y = coords[0]
    x = np.round(x)
    y = np.round(y)
    
    H = X.shape[0]
    W = X.shape[1]
    # Zero-pad image dimensions that are less than 64
    print('Initial (H,W)', (H,W))
    if H < 64 or W < 64:
        difH = np.maximum(64 - H, 0)
        difW = np.maximum(64 - W, 0)
        padH_bef = int(np.floor(difH/2))
        padH_aft = int(np.ceil(difH/2))
        difW_bef = int(np.floor(difW/2))
        difW_aft = int(np.ceil(difW/2))
        X = np.pad(X, ((padH_bef, padH_aft), (difW_bef, difW_aft)), mode='constant')
        H = X.shape[-2]
        W = X.shape[-1]
        print('Reshaped (H,W):', np.shape(X))
    
    # Crop to user-selected point center
    firstx = x - 32
    lastx = x + 32
    firsty = y-32
    lasty = y+32
    count = 0
    while firstx < 0 or lastx > W or firsty < 0 or lasty > H:
        print('count:', count)
        print(firstx)
        print(lastx)
        print(firsty)
        print(lasty)
        if firstx < 0:
            lastx -= firstx
            firstx -= firstx
        elif lastx > W:
            firstx -= (W-lastx)
            lastx -= (W-lastx)
        elif firsty < 0:
            lasty -= firsty
            firsty -= firsty
        elif lastx > H:
            firsty -= (H-lasty)
            lasty -= (H-lasty)
        count += 1
    ROI = X[firsty:lasty, firstx:lastx]
    ROI = np.reshape(ROI, (1,1,ROI.shape[-2],ROI.shape[-1])) # convert 64x64 to 1x1x64x64 ndarray
    startTime = datetime.now() # begin timer for estimating evaluation runtime
    score = classify(ROI)
    print('Runtime:', datetime.now() - startTime)

    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(X, cmap='gray')
    
    # Create a Rectangle patch
    if score>0.5:
        col = 'r'
    else:
        col = 'g'
    rect = patches.Rectangle((firstx,firsty),64,64,linewidth=2,edgecolor=col,facecolor='none')
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    plt.title('(H,W): ' + str((firstx, firsty)) + '  Score: ' + str(score))
    plt.show()    
    
    print('Prediction:', score)
    
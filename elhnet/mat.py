# -*- coding: utf-8 -*-
"""
MATLAB-like functions for Python 3.5

Created on Fri Jul 29 17:12:28 2016

@author: George
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
from skimage import data, color, img_as_float
from PIL import Image
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw() # remove annoying empty box for file dialog
#root.focus_force()

# Open file selection dialog box
def uigetfile():
    print('Select file: ')
    file_path = filedialog.askopenfilename()
    
    return file_path
    
# Open folder selection dialog box
def uigetdir():
    print('Select folder: ')
    folder_path = filedialog.askdirectory()
    
    return folder_path

# http://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
# Fill in holes
def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array
    
# Overlay color mask on greyscale image
# Adapted from: http://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
def imoverlay(img, mask, alpha=0.6, mycolor=[0,1,0]): # color=[r, g, b]
    # Construct a colour image to superimpose
    rows, cols = mask.shape
    color_mask = np.zeros((rows, cols, 3))
    mrows, mcols = np.nonzero(mask)
    for i in range(len(mrows)):
        color_mask[mrows[i],mcols[i]] = mycolor
    
    # Construct RGB version of grey-level image
    if np.ndim(img)==2:
        img_color = np.dstack((img, img, img))
    else:
        img_color = img
    
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    
    img_masked = color.hsv2rgb(img_hsv)
    
    return img_masked
    
# Print all current figures to vector PDF graphics
# http://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once
# Usage: >>>> multipage('filename.pdf')
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    


    
# Obtain 1 user-selected points coordinates on image
def pickpoint(im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap='gray')    

    # Simple mouse click function to store coordinates
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
    
#        print('x = %d, y = %d' %(ix, iy))
    
        # assign global variable to access outside of function
        global coords
        coords = []    
        coords.append((ix, iy))
    
        # Disconnect after 1 clicks
        if len(coords) == 1:
            fig.canvas.mpl_disconnect(cid)
            plt.close(1)
            
        return coords

    # Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.waitforbuttonpress()
    
    return coords
    
    
if __name__ == '__main__':
    print("Testing %s ..." % __file__)
    
    ### test uigetfile
    pic_name = uigetfile()    
    pic = plt.imread(pic_name)
#    pic /= np.max(pic)
#    pic = Image.open(pic_name).convert('L')
    print('pic:', pic)
    
#    ### test flood_fill
#    a = [[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]]
#    print(np.asarray(a))
#    print(flood_fill(a))
#    
#    ### test imoverlay
#    alpha = 0.6
#    img = img_as_float(data.camera())
#    rows, cols = img.shape
#    # Construct a colour image to superimpose
#    mask1 = np.zeros([rows, cols])
#    mask2 = np.zeros([rows, cols])
#    mask3 = np.zeros([rows, cols])
#    mask1[30:140, 30:140] = 1  # Red block
#    mask2[170:270, 40:120] = 1 # Green block
#    mask3[200:350, 200:350] = 1 # Blue block
#    color1 = [1, 0, 0]  # Red block
#    color2 = [0, 1, 0] # Green block
#    color3 = [0, 0, 1] # Blue block
#    
#    img_masked = imoverlay(img, mask1, alpha, color1)
#    img_masked = imoverlay(img_masked, mask2, alpha, color2)
#    img_masked = imoverlay(img_masked, mask3, alpha, color3)
#    
#    # Display the output
#    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
#                                      subplot_kw={'xticks': [], 'yticks': []})
#    ax0.imshow(img, cmap=plt.cm.gray)
#    ax1.imshow(mask1+mask2+mask3)
#    ax2.imshow(img_masked)
#    plt.show()
    
    ### test pickpoint
    coords = pickpoint(pic)
    x, y = coords[0]
    print('x = %d, y = %d' %(x, y))
#    
#    print('Done.')
    

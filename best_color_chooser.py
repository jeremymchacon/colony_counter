# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:00:37 2017

@author: chaco001
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
import os
from scipy import ndimage as ndi



def main(argv):

    if len(argv) == 0:
        print('must supply file argument!')
        return
    
    if not os.path.isfile(argv[0]):
        print('first argument must be path to image file')
        return
        
        
    cmap = 'binary'
    if (len(argv) > 1):
        cmap = argv[1]

    file = argv[0]       
    #file = "20170605_higlu_E_1.tif"
    cells = io.imread(file)
    
    cells_lab = skimage.color.rgb2lab(cells)
    cells_hsv = skimage.color.rgb2hsv(cells)
    
    
    f, axarr = plt.subplots(3,3)
    axarr[0,0].axis("off")
    axarr[0,0].imshow(cells[:,:,0], cmap = cmap)
    axarr[0,0].title.set_text('RGB, R')
    
    axarr[0,1].axis("off")
    axarr[0,1].imshow(cells[:,:,1], cmap = cmap)
    axarr[0,1].title.set_text('RGB, G')
    
    axarr[0,2].axis("off")
    axarr[0,2].imshow(cells[:,:,2], cmap = cmap)
    axarr[0,2].title.set_text('RGB, B')
    
    axarr[1,0].axis("off")
    axarr[1,0].imshow(cells_lab[:,:,0], cmap = cmap)
    axarr[1,0].title.set_text('LAB, L')
    
    axarr[1,1].axis("off")
    axarr[1,1].imshow(cells_lab[:,:,1], cmap = cmap)
    axarr[1,1].title.set_text('LAB, A')
    
    axarr[1,2].axis("off")
    axarr[1,2].imshow(cells_lab[:,:,2], cmap = cmap)
    axarr[1,2].title.set_text('LAB, B')
    
    axarr[2,0].axis("off")
    axarr[2,0].imshow(cells_hsv[:,:,0], cmap = cmap)
    axarr[2,0].title.set_text('HSV, H')
    
    axarr[2,1].axis("off")
    axarr[2,1].imshow(cells_hsv[:,:,1], cmap = cmap)
    axarr[2,1].title.set_text('HSV, S')
    
    axarr[2,2].axis("off")
    axarr[2,2].imshow(cells_hsv[:,:,2], cmap = cmap)
    axarr[2,2].title.set_text('HSV, V')
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
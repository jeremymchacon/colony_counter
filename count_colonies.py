# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:09:17 2016

@author: Jeremy
"""

# usage: python count_colonies.py imagename.tif

import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
import os

def round_mask(im, center, radius):
    """ sets all pixels outside a radius around the center to zero
    returns this as a new image (does not mutate im)
    """
    new_im = im.copy()
    im_size = im.shape
    for x in range(im_size[0]):
        for y in range(im_size[1]):
            if (x-center)**2 + (y-center)**2 >= radius**2:
                new_im[x,y] = 0   
    return(new_im)
    
def get_nonzero_otsu(im):
    flattened = np.ndarray.flatten(im)
    flattened = flattened[flattened > 0]
    thresh = skimage.filters.threshold_otsu(flattened)
    return(thresh)
    
def main(argv):

    if len(argv) == 0:
        print('must supply file argument!')
        return
    
    if not os.path.isfile(argv[0]):
        print('first argument must be path to image file')
        return
        
    show_plot = True
    if len(argv) > 1 and argv[1] == "no_plot":
        show_plot = False

        

    # read file    
    file = argv[0]    
    cells = io.imread(file)
    cells = skimage.img_as_float(cells)

    # change to lab, grab l
    lab = skimage.color.rgb2lab(cells)
    #plt.imshow(lab[:,:,1])
    
    # filter 
    lab1 = skimage.filters.gaussian(lab,5)

    # normalize to 0-1
    lab1 = lab[:,:,1] - np.min(lab[:,:,1])
    lab1 /= np.max(lab1)
    
    # invert
    lab1 = 1-lab1
    
    # find center and half-width
    center = lab1.shape[0] / 2
    radius = lab1.shape[0] / 2
    
    # mask dish
    masked = round_mask(lab1, center, radius)

    # apply threshold
    thresh = get_nonzero_otsu(masked)
    threshed = masked > thresh
    
    #  opened to erase teeny things
    strel = skimage.morphology.disk(7)
    opened = skimage.morphology.binary_opening(threshed, selem = strel)

    #smooth smoothed, then mask with the opened image
    smoothed = skimage.filters.gaussian(lab1,4)
    smoothed[~opened] = 0
    
    # find the local peaks
    coordinates = skimage.feature.peak_local_max(smoothed, min_distance=5)
    
    # plot
    if show_plot:
        plt.imshow(cells, cmap=plt.cm.gray)
        plt.hold(True)
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        plt.show()
    
    #coords ] 
    np.savetxt(sys.stdout, coordinates, fmt = '%01d')


    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    



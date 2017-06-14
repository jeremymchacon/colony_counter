#! /usr/bin/Rscript
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:09:17 2016

@author: Jeremy
"""

# usage: python count_colonies.py imagename.tif
# options:
#       plot -- saves the plot
#       lab 0,1,2  decide whether to use L (0) A (1) or B (2)

import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
import os
import subprocess
from scipy import ndimage as ndi
import pygame
import pandas as pd

from count_colonies_functions import *

def main(argv):

    if len(argv) == 0:
        print('count_colonies_interactive.py usage:')
        print('    count_colonies_interactive.py [filename] [optional arguments]')
        print('         e.g. count_colonies_interactive.py petridish.tif')
        print('    optional arguments:')
        print('         plot')
        print('             - shows the plot at the end of the analysis (saves regardless)')
        print('         colorspace')
        print('             - change the default colorspace (default is L channel from LAB')
        print('             - if you use colorspace, you must have two additional arguments')        
        print('             - the first must be the colorspace, a choice of either RGB, LAB, HSV')        
        print('             - the second is the channel to use in that colorspace, a choice of either 0, 1, 2')        
        print('             - e.g. count_colonies_interactive.py petridish.tif colorspace RGB 1')        
        print('                   - this will use the green channel from the RGB colorspace for the analysis')        
        print('             - if you are uncertain about the best colorspace, use the best_color_chooser.py program')        
        print('         calc_dists')
        print('             - calculates a number of distance metrics using R and the spatstat package')
        print('             - only do this if you have those installed, and Rscript is visible by your system')

        return
    
    if not os.path.isfile(argv[0]):
        print('first argument must be path to image file')
        return

    #default arguments        
    show_plot = False
    color_space = "LAB"    
    color_channel = 0
    calc_dists = False
    
    if len(argv) > 1:
        for i in range(1,len(argv)):        
            if argv[i] == "plot":
                show_plot = True
            if argv[i] == 'colorspace':
                color_space = argv[i+1]
                color_channel = int(argv[i+2])
            if argv[i] == 'calc_dists':
                calc_dists = True
        

    # read file    
    #file = "20170601_low_buffer_1.tif"
    #file = '20170605_higlu_E_1.tif'
    file = argv[0]    
    cells = io.imread(file)
    im_for_viewing = cells

    cells = skimage.img_as_float(cells)


    # change to chosen color channel
    if color_space == "LAB":
        img = skimage.color.rgb2lab(cells)
    elif color_space == "RGB":
        img = cells
    elif color_space == "HSV":
        img = skimage.color.rgb2hsv(cells)
    else:
        print("colorspace first argument must be LAB, RGB, or HSV, quitting")
        quit()
    #plt.imshow(lab[:,:,1])
    
    # filter 
    img1 = ndi.gaussian_filter(img,5)

    # normalize to 0-1
    img1 = img1[:,:,color_channel] - np.min(img1[:,:,color_channel])
    img1 /= np.max(img1)
    
    

    # click to see if colony is lighter than or darker than background
    pos = get_clicks(im_for_viewing, caption = "click on one colony then on background then exit window", window_width = 640)
    
    
    # average clicked area over 9 pixel square
    foreground = 0
    background = 0
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            foreground += img1[pos[0][0]+i, pos[0][1]+j]    
            background += img1[pos[1][0]+i, pos[1][1]+j]    
    foreground /= 9
    background /= 9
    # invert if background is brighter
    if background > foreground:
        img1 = 1-img1
    
    # find center and half-width
    center = img1.shape[0] / 2
    radius = img1.shape[0] / 2
    
    # mask dish
    circle = make_circle_selection(im_for_viewing)
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])

    masked = round_mask(img1, center, radius)

    # apply threshold
    thresh = get_nonzero_otsu(masked)
    threshed = masked > thresh

    
    #  opened to erase teeny things
    strel = skimage.morphology.disk(7)
    opened = skimage.morphology.binary_opening(threshed, selem = strel)

    #smooth smoothed, then mask with the opened image
    smoothed = ndi.gaussian_filter(img1,4)
    smoothed[~opened] = 0
    
    # find the local peaks
    coordinates = skimage.feature.peak_local_max(smoothed, min_distance=5)
    coordinates = [(x[0], x[1]) for x in coordinates]
    # let the user add additional points
    coordinates = add_or_remove_locs_with_clicks(im_for_viewing, coordinates, caption = "click to add colonies, left-shift+click to de-select")

    coordinates = np.asarray(coordinates)

    # separate connected colonies
    bw = smoothed > 0    
    distance = ndi.distance_transform_edt(bw)
    coordinates_in_im = skimage.feature.peak_local_max(smoothed, indices = False, min_distance=5)
    markers = skimage.measure.label(coordinates_in_im)
    labels_ws = skimage.morphology.watershed(-distance, markers, mask = bw)
    
    rp = skimage.measure.regionprops(labels_ws)
    
    # find the objects that don't have clicks in them
    rp_not_clicked = []
    for curr_rp in rp:
        coords = curr_rp.coords
        any_coordinate_in_object = False
        for i in range(len(coordinates)):
            if coordinates[i][0] in coords[:,0] and coordinates[i][1] in coords[:,1]:
                any_coordinate_in_object = True
                break
        rp_not_clicked.append(any_coordinate_in_object)
            
    # remove those objects from rp
    rp = [rp[i] for i in range(len(rp)) if rp_not_clicked[i] ]
    
    centroids = [rp[i].centroid for i in range(len(rp))]
    eccentricity = [rp[i].eccentricity for i in range(len(rp))]
    x = [c[1] for c in centroids]
    y = [c[0] for c in centroids]
    areas = np.asarray([rp[i].area for i in range(len(rp))])
    colony = [str(c) for c in range(len(x))]
    #plt.scatter(x, y, np.sqrt(areas))
    
    df = pd.DataFrame({'x' : x,
                       'y' : y,
                       'area' : areas,
                       'eccentricity' : eccentricity,
                       'colony' :range(len(x)),
                       'petri_x' : circle[0],
                       'petri_y' : circle[1],
                       'petri_radius' : circle[2]})
    
    # plot

    f, axarr = plt.subplots(1,2)
    axarr[0].axis("off")
    axarr[0].imshow(im_for_viewing)
    axarr[0].hold(True)
    axarr[0].plot(x, y, 'r.', markersize = 1)
    for i, txt in enumerate(colony):
        axarr[0].annotate(txt, (x[i]+9, y[i]-9), color = '#FFFFFF', size = 2)
    axarr[1].axis("off")
    axarr[1].imshow(labels_ws)
    axarr[1].hold(True)
    axarr[1].plot(x, y, 'r.', markersize = 1)
    for i, txt in enumerate(colony):
        axarr[1].annotate(txt, (x[i]+9, y[i]-9), color = '#FFFFFF', size = 2)

    plt.savefig('{}_result_img.png'.format(file),
                dpi = 300,
                bbox_inches ='tight')
      
    
    
    df.to_csv('{}_results.csv'.format(file))    
    # use R to calculate Voronoi in a circle (no python package does this!)
    if calc_dists:
        subprocess.call(["Rscript","./distance_metrics_calc.R",file])
    

    if show_plot:
        plt.show()  


    
if __name__ == '__main__':
    main(sys.argv[1:])
    


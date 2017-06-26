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
from skimage import segmentation
import os
import subprocess
from scipy import ndimage as ndi
import pygame
import pandas as pd
import math

from count_colonies_functions import *

def main(argv):

    if len(argv) == 0:
        print('count_colonies_interactive.py usage:')
        print('    count_colonies_interactive.py [filename] [optional arguments]')
        print('         e.g. count_colonies_interactive.py petridish.tif')
        print('    optional arguments:')
        print('         plot')
        print('             - shows the plot at the end of the analysis (saves regardless)')
        print('         calc_dists')
        print('             - calculates a number of distance metrics using R and the spatstat package')
        print('             - only do this if you have those installed, and Rscript is visible by your system')

        return
    
    if not os.path.isfile(argv[0]):
        print('first argument must be path to image file')
        return
    file = argv[0]

    #default arguments        
    show_plot = False
    calc_dists = False
    
    if len(argv) > 1:
        for i in range(1,len(argv)):        
            if argv[i] == "plot":
                show_plot = True
            if argv[i] == 'calc_dists':
                calc_dists = True
        
    # load file, select petri dish region
    img = io.imread(file)
    orig = img
    circle = make_circle_selection(orig)
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])

    result = get_colony_image(orig, center, radius)
    result = round_mask(result, center, radius)
    
    plt.imshow(result, cmap = "binary", interpolation = "none")

    # use result image to get a threshold and separate colony from background
    result_for_thresh = round_mask(result, center, radius*.9)
    thresh = skimage.filters.threshold_otsu(result_for_thresh[result_for_thresh > 0])
    area_image = result > thresh    
    
    plt.imshow(result > thresh, cmap = "binary", interpolation = "nearest")
    

    # to find colony peaks, search for local peaks in a image made by multiplying
    # a distance-transformed image with the result image
    distance = ndi.distance_transform_edt(area_image)    
    distance_result = distance * result
    coords = skimage.feature.peak_local_max(distance_result, min_distance = 10)
    x = [c[1] for c in coords]
    y = [c[0] for c in coords]
    plt.imshow(orig, interpolation = "nearest", cmap = "spectral")
    plt.scatter(x,y)
    
    # toss colonies on the petri dish border
    x, y = remove_coords_past_point(x,y, center, radius*0.9)
    coords = [[y[i], x[i]] for i in range(len(x))]
    
    # let the user fix the point selection
    coords = add_or_remove_locs_with_clicks(orig, coords, caption = "click to add colonies, left-shift+click to de-select")
    coords = np.asarray(coords)
    x = [c[1] for c in coords]
    y = [c[0] for c in coords]
    plt.imshow(orig, interpolation = "nearest", cmap = "spectral")
    plt.scatter(x,y)
    

    
    # separate connected colonies
    # make a coordiatnates image
    coord_img = np.zeros(area_image.shape, dtype = bool)
    for k in range(len(coords)):
        coord_img[coords[k][0], coords[k][1]] = True
        
    # using random walkers to do splitting
    markers = skimage.measure.label(coord_img)
    markers[~area_image] = -1
    rw = skimage.segmentation.random_walker(area_image, markers)
    rp = skimage.measure.regionprops(rw) 
    plt.imshow(rw, cmap = 'spectral', interpolation = "nearest")
    plt.scatter(x,y)    
    
    
    # find the objects that don't have clicks in them
    rp_clicked = []
    for curr_rp in rp:
        rp_points = curr_rp.coords
        any_coordinate_in_object = False
        for i in range(len(coords)):
            if coords[i][0] in rp_points[:,0] and coords[i][1] in rp_points[:,1]:
                any_coordinate_in_object = True
                break
        rp_clicked.append(any_coordinate_in_object)
            
    # remove those objects from rp
    rp = [rp[i] for i in range(len(rp)) if rp_clicked[i] ]
    
    
# the ordering between x/y and the regionprops objects is not the same, so fix it        
    new_order = [-1 for _ in range(len(x))]
    for j,curr_rp in enumerate(rp):
        rp_points = curr_rp.coords
        for i, coord in enumerate(coords):
            for curr_point in rp_points:
                if all(coord == curr_point):
                    new_order[j] = i
                    break
    
    x = [x[new_order[i]] for i in range(len(new_order))]
    y = [y[new_order[i]] for i in range(len(new_order))]

    plt.imshow(rw, cmap = 'spectral', interpolation = "nearest")
    plt.scatter(x,y)       

    eccentricity = [rp[i].eccentricity for i in range(len(rp))]
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
    axarr[0].imshow(orig)
    axarr[0].hold(True)
    axarr[0].plot(x, y, 'r.', markersize = 1)
    for i, txt in enumerate(colony):
        axarr[0].annotate(txt, (x[i]+9, y[i]-9), color = '#FFFFFF', size = 2)
    axarr[1].axis("off")
    axarr[1].imshow(rw)
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
    


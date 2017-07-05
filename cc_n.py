#! /usr/bin/Rscript
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:09:17 2016

@author: Jeremy
"""



import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
from skimage import segmentation
import sklearn
from sklearn import mixture
import os
import subprocess
from scipy import ndimage as ndi
import pygame
import pandas as pd
import math

from count_colonies_functions import *

argv = ["20170626_succ_1.tif", "n","2"]
def main(argv):

    if len(argv) == 0:
        print('cc_n.py usage:')
        print('    cc.py [filename] [optional arguments]')
        print('         e.g. cc.py petridish.tif')
        print('    optional arguments:')
        print('         n')
        print('             - specify number of species on dish.  default = 1')
        print('             - the argument immediately following n must be an integer > 0')        
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
    n_species = 1
    
    if len(argv) > 1:
        for i in range(1,len(argv)):        
            if argv[i] == "plot":
                show_plot = True
            if argv[i] == 'calc_dists':
                calc_dists = True
            if argv[i] == 'n':
                n_species = int(argv[i+1])
        
    # load file, select petri dish region
    img = io.imread(file)
    orig = img
    circle = make_circle_selection(orig)
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])

    result = get_colony_image(orig, center, radius)
    result = round_mask(result, center, radius)

    # use result image to get a threshold and separate colony from background
    result_for_thresh = round_mask(result, center, radius*.9)
    thresh = skimage.filters.threshold_otsu(result_for_thresh[result_for_thresh > 0])
    area_image = result > thresh    
    area_image = skimage.morphology.remove_small_objects(area_image, 32)
    # plt.imshow(area_image)
        

    # to find colony peaks, search for local peaks in a image made by multiplying
    # a distance-transformed image with the result image
    distance = ndi.distance_transform_edt(area_image)    
    distance_result = distance * result
    coords = skimage.feature.peak_local_max(distance_result, min_distance = 10)
    x = [c[1] for c in coords]
    y = [c[0] for c in coords]

    # toss colonies on the petri dish border
    x, y = remove_coords_past_point(x,y, center, radius*0.9)
    coords = [[y[i], x[i]] for i in range(len(x))]
    
    # let the user fix the point selection
    coords = add_or_remove_locs_with_clicks(np.vstack((orig,orig)), coords, caption = "click to add colonies, left-shift+click to de-select")
    coords = np.asarray(coords)
    x = [c[1] for c in coords]
    y = [c[0] for c in coords]
    x = np.array(x)
    y = np.array(y)

    
    # toss blobs lacking a click
    colony_image = remove_markerless_blobs(area_image, x, y)
    
    
    # prep stuff for single species, in case there aren't > 1 species
    single_species_centers = np.ones(len(x), dtype = bool) # changes if mult species, otherwise used as the default
    blobs = skimage.measure.label(colony_image)
    blobs[~colony_image] = 0
    blobs,_,_ = skimage.segmentation.relabel_sequential(blobs)
    rp = skimage.measure.regionprops(blobs) 
    single_species_colony_image = blobs # will change if > 1 species
    species = np.zeros(len(x))
    
    if n_species > 1:
        species_image = get_species_image(orig, center, radius)
        #get intensitieis for gaussian-mixture fitting
        I = get_intensity_at_xy(species_image, x, y)

    
        gmm = sklearn.mixture.GMM(n_species).fit(I)
        species_guess = gmm.predict(I)    
    
        # have user to click through the species labels and change them
        species_guess = change_species(np.vstack((orig,orig)), x, y, species_guess, caption = "", window_width = 1200)
 
        # figure out which are the single-species blobs
        single_species_centers, single_species_blobs = determine_which_blobs_have_multi_species(rp, x, y, species_guess)    
        
        # make two images, one the single_species image, one the multi-species image
        single_species_colony_image = np.zeros(colony_image.shape)
        multi_species_colony_image = np.zeros(colony_image.shape)
        for i in range(len(single_species_blobs)):
            label = rp[i].label
            if single_species_blobs[i]:
                single_species_colony_image[blobs == label] = label
            else:
                multi_species_colony_image[blobs == label] = label   
            
        # segment the multi-species image using a sobel filter and watershed
        multi_species_img = species_image.copy()
        multi_species_img[multi_species_colony_image == 0] = 0
        multi_species_segmentation = sobel_segment(multi_species_img, x[~single_species_centers],
                              y[~single_species_centers])    
        multi_species_segmentation[multi_species_segmentation == -1] = 0


    
    # do the single-species segmentation regardless of n_species
    colony_segmentation = random_walker_segment(single_species_colony_image > 0, 
                                x[single_species_centers],
                                y[single_species_centers])
    colony_segmentation[colony_segmentation == -1] = 0                            


    # if there were > 1 species, put it all back together 
    if n_species > 1:
        multi_species_segmentation[multi_species_segmentation > 0] = multi_species_segmentation[multi_species_segmentation > 0] + np.max(colony_segmentation)    
        colony_segmentation += multi_species_segmentation
    
        single_x = x[single_species_centers]
        single_y = y[single_species_centers]
        multi_x = x[~single_species_centers]
        multi_y = y[~single_species_centers]    
        x = np.concatenate((single_x, multi_x))
        y = np.concatenate((single_y, multi_y))
        species = np.concatenate((species_guess[single_species_centers],
                                  species_guess[~single_species_centers]))        



    # re-order everthing so it is in the right order to match the final
    # regionprops measurement
    rp = skimage.measure.regionprops(colony_segmentation)
    new_order = [-1 for _ in range(len(x))]
    for j,curr_rp in enumerate(rp):
        rp_points = curr_rp.coords
        for i in range(len(x)):
            curr_xy = np.array([y[i],x[i]])
            for curr_point in rp_points:
                if all(curr_xy == curr_point):
                    new_order[j] = i
                    break
    
    x = [x[new_order[i]] for i in range(len(new_order))]
    y = [y[new_order[i]] for i in range(len(new_order))]
    species = [species[new_order[i]] for i in range(len(new_order))]
  

    # make final measurements and save!
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
                       'petri_radius' : circle[2],
                       'species' : species})
    
    # plot
    

    f, axarr = plt.subplots(1,2)
    axarr[0].axis("off")
    axarr[0].imshow(orig)
    axarr[0].scatter(x, y, c = 1 + np.array(species), s = 2, marker = '.')
    for i, txt in enumerate(colony):
        axarr[0].annotate(txt, (x[i]+9, y[i]-9), color = '#FFFFFF', size = 2)
    axarr[1].axis("off")
    axarr[1].imshow(colony_segmentation)
    axarr[1].scatter(x, y, c = 1 + np.array(species), s = 1, marker = '.')
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
    


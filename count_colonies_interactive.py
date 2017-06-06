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
from scipy import ndimage as ndi
import pygame
import pandas as pd

def get_clicks(array, caption = "", window_width = 640):
    #array = smoothed

    # make the surface, get it's untransformed shape
    surface = pygame.surfarray.make_surface(array)
    array_size = array.shape[0:2]
    
    # figure out the transformtaion size and the ratio, to back-transform points
    ratio = float(array_size[0]) / window_width
    height = int(array_size[1] / ratio)
    
    surface = pygame.transform.scale(surface, (window_width, height))
    
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((window_width, height)) 
    screen.blit(surface, (0,0))
    pygame.display.flip()

    clicks = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # un-transform clicks
                clicks = [(int(x[0] * ratio), int(x[1] * ratio)) for x in clicks]
                return(clicks)
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                pygame.draw.circle(surface, (243,243,21), pos, 5, 1)
                pygame.draw.circle(surface, (0,0,255), pos, 4, 1)
                pygame.draw.circle(surface, (243,243,21), pos, 3, 1)

                screen.blit(surface, (0,0))
                pygame.display.flip()
                #print("user clicked at {}".format(pos))
                clicks.append(pos)
                
def add_or_remove_locs_with_clicks(array, locs, caption = "", window_width = 800):
    """
    takes an array e.g. image, shows as pygame display, plots the locs (list of
    xy tuples) on it, and adds locations to locs when clicked, which are
    returned
    """
       
    surface = pygame.surfarray.make_surface(array)
    array_size = array.shape[0:2]
    
    # figure out the transformtaion size and the ratio, to back-transform points
    ratio = float(array_size[0]) / window_width
    height = int(array_size[1] / ratio)
    
    surface = pygame.transform.scale(surface, (window_width, height))
    
    # transform locs and draw them
    locs = [(int(x[0] / ratio), int(x[1] / ratio)) for x in locs]
    for loc in locs:
        pygame.draw.circle(surface, (243,243,21), loc, 5, 1)
        pygame.draw.circle(surface, (0,0,255), loc, 4, 1)
        pygame.draw.circle(surface, (243,243,21), loc, 3, 1)    
        
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((window_width, height)) 
    screen.blit(surface, (0,0))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                locs = [(int(x[0] * ratio), int(x[1] * ratio)) for x in locs]
                return(locs)
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos

                # shift click to remove
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    # do nothing if no locations
                    if len(locs) == 0:
                        continue
                    
                    # figure out closest
                    dists = [(xy[0]-pos[0])**2 + (xy[1]-pos[1])**2 for xy in locs]
                    dists = np.asarray(dists)
                    closest = np.where(dists == np.min(dists))[0][0]
                    
                    # toss closest (if there)
                    try:
                        locs.remove(locs[closest])
                    except:
                        pass
                    surface = pygame.surfarray.make_surface(array)
                    surface = pygame.transform.scale(surface, (window_width, height))

                    for loc in locs:
                        pygame.draw.circle(surface, (243,243,21), loc, 5, 1)
                        pygame.draw.circle(surface, (0,0,255), loc, 4, 1)
                        pygame.draw.circle(surface, (243,243,21), loc, 3, 1)  
                else:
                    pygame.draw.circle(surface, (243,243,21), pos, 5, 1)
                    pygame.draw.circle(surface, (0,0,255), pos, 4, 1)
                    pygame.draw.circle(surface, (243,243,21), pos, 3, 1)
                    locs.append(pos) 

                screen.blit(surface, (0,0))
                pygame.display.flip()






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
        
    show_plot = False
    lab_color = 0
    if len(argv) > 1:
        for i in range(1,len(argv)):        
            if argv[i] == "plot":
                show_plot = True
            if argv[i] == 'lab':
                lab_color = argv[i+1]

        

    # read file    
    #file = "20170601_low_buffer_1.tif"
    file = argv[0]    
    cells = io.imread(file)
    cells = skimage.img_as_float(cells)
    
    # keep a grayscale for showing 
    gray = skimage.color.rgb2gray(cells)

    # change to lab, grab l
    lab = skimage.color.rgb2lab(cells)
    #plt.imshow(lab[:,:,1])
    
    # filter 
    lab1 = ndi.gaussian_filter(lab,5)

    # normalize to 0-1
    lab1 = lab[:,:,lab_color] - np.min(lab[:,:,lab_color])
    lab1 /= np.max(lab1)
    

    # click to see if colony is lighter than or darker than background
    pos = get_clicks(lab[:,:,lab_color], caption = "click on one colony then on background then exit window", window_width = 640)
    
    
    # average clicked area over 9 pixel square
    foreground = 0
    background = 0
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            foreground += lab1[pos[0][0]+i, pos[0][1]+j]    
            background += lab1[pos[1][0]+i, pos[1][1]+j]    
    foreground /= 9
    background /= 9
    # invert if background is brighter
    if background > foreground:
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
    smoothed = ndi.gaussian_filter(lab1,4)
    smoothed[~opened] = 0
    
    # find the local peaks
    coordinates = skimage.feature.peak_local_max(smoothed, min_distance=5)
    coordinates = [(x[0], x[1]) for x in coordinates]
    # let the user add additional points
    coordinates = add_or_remove_locs_with_clicks(lab[:,:,lab_color], coordinates, caption = "click to add colonies, left-shift+click to de-select")

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
    #plt.scatter(x, y, np.sqrt(areas))
    
    df = pd.DataFrame({'x' : x,
                       'y' : y,
                       'area' : areas,
                       'eccentricity' : eccentricity})
    
    # plot
    if show_plot:
        f, axarr = plt.subplots(1,2)
        axarr[0].axis("off")
        axarr[0].imshow(cells)
        axarr[0].hold(True)
        axarr[0].plot(x, y, 'r.', markersize = 1)
        axarr[1].axis("off")
        axarr[1].imshow(labels_ws)
        axarr[1].hold(True)
        axarr[1].plot(x, y, 'r.', markersize = 1)
        #plt.show()
        plt.savefig('{}_result_img.png'.format(file),
                    dpi = 300,
                    bbox_inches ='tight')
        
    
    
    #coords ] 
    np.savetxt(sys.stdout, df, 
               header = 'area eccentricity x y',
               fmt = '%03.3f',
               comments = '')


    
if __name__ == '__main__':
    main(sys.argv[1:])
    


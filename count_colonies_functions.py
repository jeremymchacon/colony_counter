# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:37:41 2017

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
import pygame
import pandas as pd
import math
import random

BLACK = (  0,   0,   0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)


def determine_which_blobs_have_multi_species(rp, x, y, species):
    # this takes in a regionprops object from a labeled image, and x/y coordinates of different species, 
    # based on
    # the integer vector species, where different ints mean different species.
    # it returns two things:
    #   single_species_centers, which is a boolean vector of length x, denoting
    #       whether that particular x/y location is in a blob containing only one species (True)
    #        or more than one species (False)
    #   single_species_blobs, which is a boolean vector as long as the number of 
    #       blobs in the image, denoting whether that blob contains one species (True)
    #       or more than one species (False)
    single_species_centers = np.ones(len(x), dtype = bool) # false means a click that in in a blob with other species
    single_species_blobs = np.ones(len(rp), dtype = bool) # false means a blob with >1 species
    
    for i in range(len(rp)):
        curr_blob = rp[i].coords
        n_colonies = 0
        species_in_curr_blob = []
        centers_in_curr_blob = []
        for curr_coord in curr_blob:
            for j in range(len(x)):
                if all([y[j], x[j]] == curr_coord):
                    n_colonies += 1
                    species_in_curr_blob.append(species[j])
                    centers_in_curr_blob.append(j)
        species_in_curr_blob = list(set(species_in_curr_blob))
        if len(species_in_curr_blob) > 1:
            single_species_blobs[i] = False
            single_species_centers[centers_in_curr_blob] = False   
    return((single_species_centers, single_species_blobs))
    

def get_intensity_at_xy(img, x, y):
    # returns a list of list of intensities at x-y locations in the grayscale image img
    # useful, for example, to feed into a clustering algorithm
    I = list() # stores H intensities per colony center
    for i in range(len(x)):
        rows = range((y[i]-1),(y[i]+2))
        cols = range((x[i]-1),(x[i]+2))
        all_dat = [np.mean(img[rows,cols], 0)]
        I.append(all_dat)
    return(I)
    
def make_marker_image(img, x, y, bg = -1):
    # makes a marker image the same size as img, for use in segmentation with watershed or random walkers
    # labels the bg as bg, -1 by default
    coord_img = np.zeros(img.shape, dtype = bool)
    for k in range(len(x)):
        coord_img[y[k], x[k]] = True
    markers = skimage.measure.label(coord_img)
    markers[img == 0] = bg
    return(markers)
    
    
def remove_markerless_blobs(img, x, y):
    # in the binary image img, remove blobs that don't contain an x/y (col/row) point
    markers = make_marker_image(img, x, y, -1)
    rw = skimage.segmentation.random_walker(img, markers)
    return(rw > 0)

def sobel_segment(colony_image, x, y):
    # segments a gray-scale image (colony image) using the x, y locations
    # colony_image should look different depending on species
    markers = make_marker_image(colony_image, x, y)
    gradient = skimage.filters.sobel(colony_image)
    ws = skimage.morphology.watershed(gradient, markers)
    return(ws)

def random_walker_segment(colony_image, x, y):
    # segments a binary image (colony_image) using the x, y locations
    markers = make_marker_image(colony_image, x, y)
    rw = skimage.segmentation.random_walker(colony_image, markers)
    return(rw)

def get_colony_image(orig, center = None, radius = None):
    # this function sums a lot of abs(median-subtracted images )which puts together
    # an image in which colonies are brighter than the background, even if they
    # were darker originally.  It optionally masks the images with a round mask
    # using center = (x,y) and radius = int
    result = np.zeros(orig[:,:,0].shape)

    for j in range(5):
        if j == 0:
            img = skimage.color.rgb2xyz(orig)
        elif j == 1:
            img = skimage.color.rgb2hsv(orig)
        elif j == 2:
            img = skimage.color.rgb2lab(orig)
        elif j == 3:
            img = skimage.color.rgb2hed(orig)
        elif j == 4:
            img = skimage.color.rgb2luv(orig)
            
    
        for i in range(3):
            H = img[:,:,i]
            H = H - np.min(H)
            H /= np.max(H)
            H = ndi.gaussian_filter(H, 2)               
            
            if center is not None:               
                H = round_mask(H, center, radius)
            im_median = np.median(H[H > 0])
            
            H2 = np.abs(H - im_median)
            H2 = H2 - np.min(H2)
            H2 /= np.max(H2)
            result += H2    
    return(result)

def get_species_image(orig, center = None, radius = None):
    # this function sums a lot of (median-subtracted images )which puts together
    # an image in which colonies are different than the background. 
    # It optionally masks the images with a round mask
    # using center = (x,y) and radius = int
    result = np.zeros(orig[:,:,0].shape)

    for j in range(5):
        if j == 0:
            img = skimage.color.rgb2xyz(orig)
        elif j == 1:
            img = skimage.color.rgb2hsv(orig)
        elif j == 2:
            img = skimage.color.rgb2lab(orig)
        elif j == 3:
            img = skimage.color.rgb2hed(orig)
        elif j == 4:
            img = skimage.color.rgb2luv(orig)
            
    
        for i in range(3):
            H = img[:,:,i]
            H = H - np.min(H)
            H /= np.max(H)
            H = ndi.gaussian_filter(H, 2)               
            
            if center is not None:               
                H = round_mask(H, center, radius)
            im_median = np.median(H[H > 0])
            
            H2 = H - im_median
            H2 = H2 - np.min(H2)
            H2 /= np.max(H2)
            result += H2    
    return(result)

def remove_coords_past_point(x,y, center, radius):
    # looks through the x,y coordinates, and removes any that are radius distance 
    # from the center point
    colonies_to_toss = []
    for i in range(len(x)):
        dist = get_distance(x[i],y[i],center[0],center[1])
        if dist > radius:
            colonies_to_toss.append(i)
    x = [x[i] for i in range(len(x)) if i not in colonies_to_toss ]
    y = [y[i] for i in range(len(y)) if i not in colonies_to_toss ]
    return((x,y))


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

def change_species(array, x, y, species, caption = "", window_width = 1200):
    """
    takes an array e.g. image, shows as pygame display, plots the locs (list of
    xy tuples) on it with color defined by species.  if a user clicks near a loc,
    it changes to the next species. this continues until the user exits, at which
    point the updated species vector is returned
    """
        
    #array = orig
    colors = [BLUE, GREEN, CYAN, RED]
    species = np.array(species, dtype = int)
    n_species = len(np.unique(species))
    
    if n_species > len(colors):
        more_colors_needed = n_species - len(colors)
        for i in range(more_colors_needed):
            colors.append((random.randint(0,255), 
                           random.randint(0,255),
                           random.randint(0,255)))
    elif n_species < len(colors):
        colors = colors[0:n_species]
       
    surface = pygame.surfarray.make_surface(array)
    array_size = array.shape[0:2]
    
    # figure out the transformtaion size and the ratio, to back-transform points
    ratio = float(array_size[0]) / window_width
    height = int(array_size[1] / ratio)
    
    surface = pygame.transform.scale(surface, (window_width, height))
    
    locs = [(int(y[i] / ratio), int(x[i] / ratio)) for i in range(len(x))]    
    
    # transform locs and draw them
    for i,loc in enumerate(locs):
        pygame.draw.circle(surface, colors[species[i]], loc, 5, 1)
 
        
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((window_width, height)) 
    screen.blit(surface, (0,0))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # see if need to fix species
                return(species)
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                    
                # figure out closest
                dists = [(xy[0]-pos[0])**2 + (xy[1]-pos[1])**2 for xy in locs]
                dists = np.asarray(dists)
                closest = np.where(dists == np.min(dists))[0][0]
                
                # change species
                species[closest] += 1
                if species[closest] >= n_species:
                    species[closest] = 0

                surface = pygame.surfarray.make_surface(array)
                surface = pygame.transform.scale(surface, (window_width, height))

                for i,loc in enumerate(locs):
                    pygame.draw.circle(surface, colors[species[i]], loc, 5, 1)

                screen.blit(surface, (0,0))
                pygame.display.flip()

                
def add_or_remove_locs_with_clicks(array, locs, caption = "", window_width = 1200):
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


def make_circle_selection(array,
                         window_width = 640, 
                         caption = "left click to move, right click to change size"):
    # make_circle_selction takes in an image (array) and two optional values
    # It then displays a circle on the image, which you can move with left click
    # and change size with right click.
    # return:  
    #   (circle_center.x, circle_center.y, circle.radius)
    surface = pygame.surfarray.make_surface(array)
    array_size = array.shape[0:2]

    # figure out the transformtaion size and the ratio, to back-transform points
    ratio = float(array_size[0]) / window_width
    height = int(array_size[1] / ratio)    
    surface = pygame.transform.scale(surface, (window_width, height))
    
    # initialize the pygame rect that will be used to select under-the-hood
    rect_width = min([window_width, height]) / 2
    rect = pygame.Rect(rect_width/2, rect_width/2, rect_width, rect_width)
    
    # flags to determining what the mouse is doing
    selected_to_move = False # left click moves, right click shrinks/grows
    selected_to_resize = False # right click shirnks / grows
    
    # initialize the pygame window and show stuff
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((window_width, height)) 
    screen.blit(surface, (0,0))
    pygame.draw.rect(screen, BLACK, rect, 3)
    pygame.draw.circle(screen, BLACK, rect.center, int(rect.width / 2.),  3)
    pygame.display.flip()
    
    # the main loop, where mouse motion is detected and the rect is moved / sized accordingly
    editing_circle = True
    while editing_circle:
        
        # display the changes 
        screen.fill(BLACK)
        screen.blit(surface, (0,0))
        pygame.draw.rect(screen, BLACK, rect, 3)
        pygame.draw.circle(screen, BLACK, rect.center, int(rect.width / 2.),  3)
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                editing_circle = False
            
            # initiate a drag or resize
            if event.type == pygame.MOUSEBUTTONDOWN:
                    
                dx = rect.centerx - event.pos[0] # a
                dy = rect.centery - event.pos[1] # b
                distance_square = dx**2 + dy**2 # c^2
    
                if distance_square <= (rect.width/2.)**2: # c^2 <= radius^2
                    # left mouse (button 1) is for dragging
                    if event.button == 1:
                        selected_to_move = True
                    # right mouse (button 3) for resizing
                    elif event.button == 3:
                        selected_to_resize = True
                        curr_centerx = rect.centerx
                        curr_centery = rect.centery
                    selected_offset_x = rect.x - event.pos[0]
                    selected_offset_y = rect.y - event.pos[1]
            # end a drag or resize.  first condition for the case where the rect wasn't selected during the click
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    selected_to_move = False
                elif event.button == 3:
                    selected_to_resize = False               
            elif event.type == pygame.MOUSEMOTION and selected_to_move:
                # move object
                rect.x = event.pos[0] + selected_offset_x
                rect.y = event.pos[1] + selected_offset_y
            elif event.type == pygame.MOUSEMOTION and selected_to_resize:
                dx = curr_centerx - event.pos[0] # a
                dy = curr_centery - event.pos[1] # b
                curr_distance_square = dx**2 + dy**2 
                # this hack implements resizing in a somewhat natural way
                if curr_distance_square < distance_square:
                    rect.width -= 3
                    rect.height -= 3
                else:
                    rect.width += 3
                    rect.height += 3
                distance_square = curr_distance_square # c^2

    
    result = (rect.centerx * ratio, rect.centery * ratio, rect.width/2 * ratio)
    return(result)
        



def round_mask(im, center, radius):
    """ sets all pixels outside a radius around the center to zero
    returns this as a new image (does not mutate im)
    """

    if isinstance(center, int):
        center = (center, center)
    
    new_im = im.copy()
    im_size = im.shape
    for x in range(im_size[0]):
        for y in range(im_size[1]):
            if (x-center[0])**2 + (y-center[1])**2 >= radius**2:
                new_im[x,y] = 0   
    return(new_im)
    
def get_nonzero_otsu(im):
    flattened = np.ndarray.flatten(im)
    flattened = flattened[flattened > 0]
    thresh = skimage.filters.threshold_otsu(flattened)
    return(thresh)
    
def get_distance(x1,y1,x2,y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return(dist)
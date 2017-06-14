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

BLACK = (  0,   0,   0)


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
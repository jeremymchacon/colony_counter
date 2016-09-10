# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:49:07 2016

@author: Jeremy
"""


import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
import os
from scipy import ndimage as ndi

def show_clicks_on_image(array, locs):
    plt.imshow(array, cmap=plt.cm.gray)
    plt.hold(True)
    plt.plot(locs[:, 1], locs[:, 0], 'r.')
    plt.show()

def show_image(array):
    """
    Takes an array, e.g. a 2 or 3d image from skimage, and shows it as a pygame
    display
    """
    surface = pygame.surfarray.make_surface(array)
    pygame.init()
    screen = pygame.display.set_mode(array.shape[0:2])
    screen.blit(surface, (0,0))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

def get_xy_of_click_on_image(array):
    """
    Takes an array, e.g. a 2 or 3d image from skimage, and shows it as a pygame
    display, waits for a click from the user, and returns the xy of the click 
    as a tuple
    """
    surface = pygame.surfarray.make_surface(array)
    pygame.init()
    screen = pygame.display.set_mode(array.shape[0:2])
    screen.blit(surface, (0,0))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return((None, None))
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                print("user clicked at {}".format(pos))
                pygame.quit()                
                return(pos)

def get_xy_of_clicks_on_image(array):
    """
    Takes an array, e.g. a 2 or 3d image from skimage, and shows it as a pygame
    display, waits for clicks from the user, and returns the xy of the clicks 
    as a list of tuples when the user exits the window
    """
    surface = pygame.surfarray.make_surface(array)
    pygame.init()
    screen = pygame.display.set_mode(array.shape[0:2])
    screen.blit(surface, (0,0))
    pygame.display.flip()
    clicks = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return(clicks)
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                pygame.draw.circle(surface, (243,243,21), pos, 5, 1)
                pygame.draw.circle(surface, (0,0,255), pos, 4, 1)
                pygame.draw.circle(surface, (243,243,21), pos, 3, 1)

                screen.blit(surface, (0,0))
                pygame.display.flip()
                print("user clicked at {}".format(pos))
                clicks.append(pos)


def add_locs_with_clicks(array, locs):
    """
    takes an array e.g. image, shows as pygame display, plots the locs (list of
    xy tuples) on it, and adds locations to locs when clicked, which are
    returned
    """
    surface = pygame.surfarray.make_surface(array)
    pygame.init()
    screen = pygame.display.set_mode(array.shape[0:2])
    for loc in locs:
        pygame.draw.circle(surface, (243,243,21), loc, 5, 1)
        pygame.draw.circle(surface, (0,0,255), loc, 4, 1)
        pygame.draw.circle(surface, (243,243,21), loc, 3, 1)    
        
    screen.blit(surface, (0,0))
    pygame.display.flip()
    clicks = []


    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return(locs + clicks)
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                pygame.draw.circle(surface, (243,243,21), pos, 5, 1)
                pygame.draw.circle(surface, (0,0,255), pos, 4, 1)
                pygame.draw.circle(surface, (243,243,21), pos, 3, 1)

                screen.blit(surface, (0,0))
                pygame.display.flip()
                print("user clicked at {}".format(pos))
                clicks.append(pos)
    

def add_or_remove_locs_with_clicks(array, locs):
    """
    takes an array e.g. image, shows as pygame display, plots the locs (list of
    xy tuples) on it, and adds locations to locs when clicked, which are
    returned
    """
    surface = pygame.surfarray.make_surface(array)
    pygame.init()
    screen = pygame.display.set_mode(array.shape[0:2])
    for loc in locs:
        pygame.draw.circle(surface, (243,243,21), loc, 5, 1)
        pygame.draw.circle(surface, (0,0,255), loc, 4, 1)
        pygame.draw.circle(surface, (243,243,21), loc, 3, 1)    
        
    screen.blit(surface, (0,0))
    pygame.display.flip()


    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return(locs)
            if event.type == pygame.MOUSEBUTTONUP:
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
                
 
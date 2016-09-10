# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:35:44 2016

@author: Jeremy
"""

import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from skimage import feature
import os
from scipy import ndimage as ndi

class imageanalysis():
    # holds image, processes it
    def __init__(self, path):
        self.set_image_from_path(path)
        
        self.original_image = self.image
        self.coordinates = []
        
    def set_image_from_path(self, path):
        self.image = io.imread(path)    

        self.image = skimage.img_as_float(self.image)

        # change to lab, grab l
        self.image = skimage.color.rgb2lab(self.image)
        self.image = self.image[:,:,1]
        
    def reset_image(self):
        self.image = self.original_image
        
    def detect_colonies(self):
        self.make_bw()
        self.open_image()
        self.smooth_original()
        self.find_peaks()
    
    def make_bw(self):
        self.image= ndi.gaussian_filter(self.image,5)    
        self.normalize()        
        self.invert()       
        self.mask_with_circle()  
        self.otsu_thresh()
        

        
        
    def normalize(self):
        #norm 0-1
        self.image = self.image - np.min(self.image)
        self.image /= np.max(self.image)
        
    def invert(self):
        self.image = 1-self.image
        
    def mask_with_circle(self):
        # find center and half-width
        center = self.image.shape[0] / 2
        radius = self.image.shape[0] / 2
        
        # mask dish
        im_size = self.image.shape
        for x in range(im_size[0]):
            for y in range(im_size[1]):
                if (x-center)**2 + (y-center)**2 >= radius**2:
                    self.image[x,y] = 0  
                    
    def otsu_thresh(self):
        flattened = np.ndarray.flatten(self.image)
        flattened = flattened[flattened > 0]
        thresh = skimage.filters.threshold_otsu(flattened)
        self.image = self.image > thresh  
        
    def open_image(self):
        strel = skimage.morphology.disk(7)
        self.image = skimage.morphology.binary_opening(self.image, selem = strel)

    def smooth_original(self):
        
        #smooth smoothed, then mask with the opened image
        self.original_image = ndi.gaussian_filter(self.original_image,4)
        self.original_image[~self.image] = 0
        
    def find_peaks(self):
    
        # find the local peaks
        self.coordinates = skimage.feature.peak_local_max(self.original_image, min_distance=5)
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:13:05 2016

@author: Jeremy
"""
import os
import gui_functions as gf
import image_processing_functions as imp
import sys
from skimage import io
import numpy as np
#file = 'bottomleft250_20150904.tif'

def main(argv):
    print(argv)
    if len(argv) == 0:
        print('must supply file argument!')
        return
    
    if not os.path.isfile(argv[1]):
        print('first argument must be path to image file')
        return    
    
    file = argv[1]
    print('opening {}'.format(file))
    #cells = io.imread(file)    
    #clicks = gf.add_or_remove_locs_with_clicks(cells, [(100,100),(50,300)])
    image = imp.imageanalysis(file)
    image.detect_colonies()
    gf.show_image(np.array(image.original_image, dtype="int"))
    print(image.coordinates)
    gf.show_clicks_on_image(image.original_image,image.coordinates)
    
    
if __name__ == '__main__':
    
    main(sys.argv)
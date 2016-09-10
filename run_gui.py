# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:13:05 2016

@author: Jeremy
"""

import gui_functions as gf
import sys
from skimage import io

file = 'bottomleft250_20150904.tif'

def main(argv):
    cells = io.imread(file)    
    clicks = gf.add_or_remove_locs_with_clicks(cells, [(100,100),(50,300)])

if __name__ == '__main__':
    main(sys.argv)
Colony_Counter

By: Jeremy M. Chacon
contact:  chaco001 at you em en .edu

Colony_Counter is a repo for counting bacterial colonies using python and R. 

There are two programs in this repo, both for use at the command line.

The first is count_colonies_interactive.py

Usage for count_colonies_interactive.py:

    python count_colonies_interactive <image_file> <optional args>

    e.g. python count_colonies_interactive petri_dish.tif

count_colonies_interactive.py counts colonies in an RGB image, measures their area, and their x/y. It uses
some minimal interactivity to allow the user to select the circular region within which to measure,
and to select / deselect colonies the program detects. 

Two outputs are saved in the same directory as the image:

 <image_file>_result_img.png
 <image_file>_results.csv

The first is the image with the colony locations and the colony numbers shown on the left, and the areas of these colonies on the right.

The second are the results themselves. 

Optionally, it uses R to calculate some distance-related metrics (e.g. Voronoi areas). 

count_colonies_interactive optional arguments:
    plot    	: if given, shows the plot in addition to saving it
    calc_dists	: if given, the R script distance_metrics_calc.R is used to calculate distance metriccs, which are added to the .csv
    color_space : if given, two additional arguments must be given in order:
		:	<colorspace>, which is either LAB, RGB, or HSV
		:	<channel>, which is either 0, 1, or 2
		:  together, these three optional arguments are used to select which colorspace channel the program uses to find colonies. 
		:  by default the program uses the L channel of LAB, i.e. color_space LAB 0
	

Requires:
	python: 
		numpy, matplotlib, scipy, skimage, pygame, pandas
	R (only if you want to calculate distance metrics):
		spatstat


The second program is best_color_chooser.py

Usage for best_color_chooser.py:

    python best_color_chooser.py <image_file> <optional arg>

This simple program shows each channel of the image in RGB, LAB, and HSV colorspace. 
This is useful to do before running count_colonies_interactive, so that you can use the best channel.

The optional argument is the colormap that the program uses to display the images.  By default, it uses a grayscale colormap.  You can choose any colormap available to matplotlib.imshow. Putting gibberish in the optional argument will give an error, but will also show you the list of options.

		





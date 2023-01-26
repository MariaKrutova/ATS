#!/usr/bin/env python
# coding: utf-8
# load standard libraries
"""
Created on Thu Dec  3 14:47:02 2020

@author: Maria Krutova (Maria.Krutova@uib.no)
"""

import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

import scipy.ndimage as scimg

# load custom functions
from helper_functions import load_image
from processing_functions import ATS, apply_threshold

#%% LOAD DATA
# data source: lidar or netcdf
source = sys.argv[1] if len(sys.argv)>1 else 'lidar'

# source data type: img or data file
s_type = sys.argv[2] if len(sys.argv)>2 else 'data'

# path to the input data
input_path = './examples'

# load data
if s_type == 'image' or s_type == 'img':
    Z = load_image('{}/{}_img.png'.format(input_path, source))

if s_type == 'data':
    data = np.load('{}/{}_data.npy'.format(input_path, source))
    # test data only
    beam_range, azimuth, x, y, Vr_clean, Vr_orig = data
    
    # Choose between original (Vr_orig) and cleaned (Vr_clean) data. 
    # Original data may have outliers that skew the threshold identification.
    # Cleaning function for the outliers is not provided
    Z = Vr_clean

# normalize data to the range of [0,1]
I = (Z-np.nanmin(Z))/(np.nanmax(Z)-np.nanmin(Z)) 

#%% FIND THE THRESHOLD USING THE ATS METHOD
### Parameters for the ATS method
# n-point average
n          = 4      

## Tail-fitting
# Set flag whether to perform a curve fit to smooth the histogram tail 
# and improve the threshold identification (highly recommended for a noisy tail).
fitting    = True    

# Select between tail fits:
#  - hyperbolic F(K)=a+b/K^m 
#  - cubic F(k) = p0 + p1*K + p2*K^2 + p3*K^3 
# The cubic fit is more strict.
function   = 'hyper' 

# Power of the hyperbolic fit in F(K)=a+b/K^m. 
# Higher m moves thresholds to the left (relaxes thresholds).
# Not required for the cubic fit.
m          = 5     

# Value, below which the tail is considered a flat tail.
# Increase in the case of a noisy tail to return stricter threshold  .
# If set to False or zero, the ATS method will select global extrema for both derivatives
# -- may lead to a relaxed threshold
flatness   = 0

# Set flag whether to plot the threshold search plots
draw_plots = True    

# Call the ATS method
Topt1, Topt2, = ATS(I, n, m, fit=fitting, function=function, plots=draw_plots, flatness = flatness)

# Relax the thresholds by taking their average
Topt = (Topt1 + Topt2)/2 
    
print('Thresholds:\nFrom the first derivative: {:.2f}\nFrom the second derivative: {:.2f}\nAveraged (final): {:.2f}'.format(Topt1, Topt2, Topt))    

#%% PLOTS
I2 = apply_threshold(I, Topt)

I_label = I2.copy()

# merge NaN with the background
# Assumes label 0 as a background. Comment this line if the assumption is incorrect
I_label[np.isnan(I)]=0   

# split detected object into continuous shapes
labels, n_parts = scimg.label(I_label)

# define a special color map for the shapes
bounds_lbl = np.arange(np.nanmax(labels))
cmap_labels = cm.get_cmap('nipy_spectral_r',lut=len(bounds_lbl)+1)
cmap_lbl_bounds = np.arange(len(bounds_lbl)+2) - 0.5
ticks = np.linspace(0.5, n_parts, 5)
ticks_labels = ['{:.0f}'.format(np.round(i)) for i in ticks]
extent = None

# define a special colormap to plot thresholded images -- will mark missing data
bounds_thrhld = np.array([.5, .9])
cmap_wake = cm.get_cmap('binary_r',lut=len(bounds_thrhld)+1)
cmap_bounds = np.arange(len(bounds_thrhld)+2) - 0.5

# corrections to place the colorbars nicely
l, w = np.shape(I)
if w/l>2:
    orientation='horizontal'
else:
    orientation = 'vertical'
    
# plot all derivated matrices
fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2, sharex = True, sharey=True, constrained_layout=True)

if s_type == 'image' or s_type == 'img':
    im1 = ax1.matshow(Z,  cmap=cm.coolwarm_r)   
    im2 = ax2.matshow(I,  cmap=cm.binary_r) 
    im3 = ax3.matshow(I2, cmap=cmap_wake)
    im4 = ax4.matshow(labels, cmap=cmap_labels)
    
if s_type == 'data':
    im1 = ax1.pcolormesh(x, y, Z,  cmap=cm.coolwarm_r,   shading = 'auto')   
    im2 = ax2.pcolormesh(x, y, I,  cmap=cm.binary_r,     shading = 'auto')   
    im3 = ax3.pcolormesh(x, y, I2, cmap=cmap_wake,       shading = 'auto')   
    im4 = ax4.pcolormesh(x, y, labels, cmap=cmap_labels, shading = 'auto')      
    
ax1.set_title('(a) Original data')
cbar = fig.colorbar(im1, ax = [ax1], orientation=orientation, aspect = 30)
  
ax2.set_title('(b) Normalized data')
cbar = fig.colorbar(im2, ax = [ax2], orientation=orientation, aspect = 30, label = 'Grayscale intensity')

ax3.set_title('(c) Thresholded image')
cbar = fig.colorbar(im3, ax = [ax3], ticks=[.15, .5, .85], orientation=orientation, aspect = 30)
cbar.ax.tick_params(size=0)
if w/l>2:
    cbar.ax.set_xticklabels(['background', 'no data', 'object']);
else:
    cbar.ax.set_yticklabels(['background', 'no data', 'object']);

ax4.set_title('(d) Detected shapes')
cbar = fig.colorbar(im4, ax = [ax4], ticks=ticks, orientation = orientation, label='Identified shapes')
cbar.ax.tick_params(size=0)
if w/l>2:
    cbar.ax.set_xticklabels(ticks_labels);
else:
    cbar.ax.set_yticklabels(ticks_labels);
for ax in (ax1,ax2,ax3,ax4):
    ax.axis('off')
    if w/l>=0.5 and w/l<=2:
        ax.axis('scaled')

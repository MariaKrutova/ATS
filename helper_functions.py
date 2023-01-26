"""
Created on Thu Dec  3 14:47:02 2020

@author: Maria Krutova (Maria.Krutova@uib.no)
"""
import numpy as np
from numpy import pi, sin, cos
import os

from skimage import io, img_as_float

#%%
def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:]-ret[:-n]
    return ret[n-1:]/n

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v/norm

#%% distances
# distance to concave of a function y(x)
def distance_to_concave(x,y):
    points=np.stack((x,y),1)
    p1=np.array([x[0], y[0]])
    p2=np.array([x[-1], y[-1]])
    D=np.array([np.cross(p1 - p2, p - p1) / np.linalg.norm(p1 - p2) for p in points])
    return abs(D)

# distances between points p1 and p2
def cartesian_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def polar_distance(p1, p2):
    return np.sqrt(p1[0]**2 + p2[0]**2 - 2*p1[0]*p2[0]*cos(p1[1]-p2[1]))

#%% coordinates conversion
def pol2cart(r, phi):
    if len(np.shape(r))>1: # matrix multiplication
        x = np.dot(r.T, sin(phi))
        y = np.dot(r.T, cos(phi))
    else: # compatibility with numbers
        x = r*sin(phi)
        y = r*cos(phi)
    return (x, y)

def cart2pol(x, y):
    r   = np.sqrt(x**2+y**2)
    phi = np.arctan2(x,y)
    try:
        phi[x<0]=2*pi+phi[x<0]
    except TypeError:
        if x<0:
            phi=2*pi+phi
    return (r, phi)

#%% Search for indexes

# return indexes of N lowest elements
def argmin_N(data, N=2):
    if N>len(data):
        N=len(data)
    minimums = sorted([*enumerate(data)], key=lambda x: x[1])[0:N]
    idx = [i[0] for i in minimums]
    return idx

# return indexes of N highest elements
def argmax_N(data, N=2):
    if N>len(data):
        N=len(data)
    maximums = sorted([*enumerate(data)], key=lambda x: x[1])[-2:]
    idx = [i[0] for i in maximums]
    return idx[::-1]

# Find index of [data] array closest to [element]
def find_index(data, element):
    return (abs(data-element)).argmin() 

#%% Check if directory exists. If no, create it.
def check_dir(dirName, verbose=False):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        if verbose: 
            print("Directory " , dirName ,  " Created ")
    else:  
        if verbose:   
            print("Directory " , dirName ,  " already exists") 
            
def load_image(file_name):
    image = io.imread(file_name)
    mask=np.array(1-img_as_float(image)[:,:,3], dtype=bool)
    I = img_as_float(image)[:,:,0]
    I[mask]=np.nan
    return I            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:47:02 2020

@author: Maria Krutova (Maria.Krutova@uib.no)
"""

import numpy as np
import scipy.optimize as so
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

from helper_functions import moving_average, distance_to_concave, normalize, find_index


def hyper(x, a, b, m):
    # hyperbolic function for the tail fit
    return a + b/x**m

def cubic(x, p0, p1, p2,p3):
    # cubic function for the tail fit
    p = (p0, p1, p2,p3)
    R=0
    for i in range(len(p)):
        R += p[i]*x**i
    return R    

def fit_tail(fit_func, x, y, p0, bounds):
    # fit function to the tail
    sigma=np.ones_like(y)
    sigma[[0,-1]]=1e-2
    param, c=so.curve_fit(fit_func, x, y, p0, sigma=sigma, bounds=bounds)
    distances=distance_to_concave(x, fit_func(x, *param))
    return param, distances

def cut_tail(K, H, flatness):
    # identify the tail to fit
    return K[abs(H[K])>flatness][-1]

def ATS(I, n=4, m=5, fit=True, function='hyperbolic', plots=False, verbose = False, flatness = 0.05):
    hist, K = np.histogram(I[~np.isnan(I)], bins=100, density=True)
    dk=K[1]-K[0]
    H=np.cumsum(hist)*dk
    
    x1=K[n//2:-n//2]
    x2=K[n-1:-n]
    H1=moving_average(np.gradient(H,  dk), n)
    H2=moving_average(np.gradient(H1, dk), n)
    H1_norm=normalize(H1)
    H2_norm=normalize(H2)
    
    Tmaxs_all=argrelextrema(H1_norm, np.greater)[-1]
    Tmins_all=argrelextrema(H2_norm, np.less)[-1]
    
    if isinstance(flatness, float) and flatness!=0:        
        Tmax = cut_tail(Tmaxs_all, H1_norm, flatness)
        Tmin = cut_tail(Tmins_all, H2_norm, flatness)
    else:    
        Tmax=np.argmax(H1_norm)
        Tmin=np.argmin(H2_norm)            

    if 'hyper' in function:
        ini_par = np.array([0, 1, m])
        fit_func = hyper   
        bounds = [ (-np.inf, -np.inf, m-1e-10), (np.inf, np.inf, m+1e-10) ] # workaround to fix m for the curve_fit
    elif function == 'cubic':
        ini_par = np.repeat(1, 4)
        fit_func = cubic
        bounds = [ np.repeat(-np.inf, len(ini_par)), np.repeat(np.inf, len(ini_par)) ]
    else:
        print('WARNING: The fitting function for the tail is provided incorrectly. State either \'hyper\' or \'cubic\' to prevent this message. The procedure will continue as if the hyperbolic fit of a+b*K^{} was selected.'.format(m))
        ini_par = np.array([0, 1, m])
        fit_func = hyper   
        bounds = [ (-np.inf, -np.inf, m-1e-10), (np.inf, np.inf, m+1e-10) ]
        
    if fit:
        try:
            H1_fit, D1 = fit_tail(fit_func, x1[Tmax:], H1_norm[Tmax:], ini_par, bounds)
            H2_fit, D2 = fit_tail(fit_func, x2[Tmin:], H2_norm[Tmin:], ini_par, bounds)
        except TypeError:
            print('Error: Cannot fit the function to the tail. Please, check the histogram data. The procedure will continute without smoothing the tail. May estimate the threshold incorrectly.')
            D1=distance_to_concave(x1[Tmax:], H1_norm[Tmax:])
            D2=distance_to_concave(x2[Tmin:], H2_norm[Tmin:])
            fit = False
    else:
        D1=distance_to_concave(x1[Tmax:], H1_norm[Tmax:])
        D2=distance_to_concave(x2[Tmin:], H2_norm[Tmin:])
    
    Topt1=x1[Tmax+np.argmax(D1)]
    Topt2=x2[Tmin+np.argmax(D2)]
    
    if plots:
        # plot derivatives and thresholds of requested
        plt.figure()
        plt.plot(x1, H1_norm, color='b', label = 'First derivative, H\'(K)')
        plt.plot(x2, H2_norm, color='k', label = 'Second derivative, H\"(K)')
        plt.plot(x1[Tmaxs_all], H1_norm[Tmaxs_all], 'b.', zorder=0, label = 'H\'(K) local maximums')
        plt.plot(x2[Tmins_all], H2_norm[Tmins_all], 'k.', zorder=0, label = 'H\"(K) local minimums')
        if fit:
            plt.plot(x1[Tmax:], fit_func(x1[Tmax:], *H1_fit), 'r-', lw=.75, label = 'Tail fit')
            plt.plot(x2[Tmin:], fit_func(x2[Tmin:], *H2_fit), 'r-', lw=.75)
            plt.plot([Topt1, Topt2], [fit_func(Topt1, *H1_fit), fit_func(Topt2, *H2_fit)], 
                     'ro', label = 'Thresholds')
        else:
            plt.plot([Topt1, Topt2], [H1_norm[find_index(x1, Topt1)], H2_norm[find_index(x2, Topt2)]], 
                     'ro', label = 'Thresholds')            
        
        plt.plot([x1[Tmax], x1[-1]], [H1_norm[Tmax],H1_norm[-1]], 'g--', lw = 1)
        plt.plot([x2[Tmin], x2[-1]], [H2_norm[Tmin],H2_norm[-1]], 'g--', lw = 1, label = 'Helper lines')
        plt.plot([0,1], [0,0], 'k-', lw=.5) 
        plt.xlim([0,1])
        plt.xlabel('Thresholds, $K$')
        plt.ylabel('H\'($K$) and H\"($K$)')
        plt.legend(loc = 'lower left', bbox_to_anchor=(0,-0.5), ncol = 2)         
        
    return Topt1, Topt2

def apply_threshold(I, th, fill_nan = True):
    
    I2=np.full_like(I, np.nan)
    I2[I<th]=0
    I2[I>=th]=1
    if fill_nan:
        I2[np.isnan(I2)]=0.5
    return I2  
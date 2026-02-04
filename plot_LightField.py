#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:12:08 2025

@author: wkmills
"""

import numpy as np 
import matplotlib.pyplot as plt 
#import cv2 
import os 
import sys 


def plot_bfp_centered(data_file, NA, title, crop=True, outline_NA = False, log_plot = False, save=False, transpose=False):
    
    match NA:
        case '1.3':
            (x0, y0), (w, h) = ((112,112),(801,801))
            lim = 1.3
        case '1.25':
            (x0, y0), (w, h) = ((435,435),(160,160)) 
            lim = 1.25 
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = x0+2, max_rows = h, usecols = tuple(range(y0+1, y0+w+1)))
    radius = (h-1)/2
    shape = np.shape(data) 
    print(shape) 
    circle_row = (shape[1]-1)/2
    circle_col = (shape[0]-1)/2 
    
# =============================================================================
#     # Add the other dataset 
#     data_2_location = os.getcwd() + '/LED=8mW-405sp-100x_sample(520nm-ref)_450lp-xpol-1sec_bfp.csv'
#     data_2 = np.loadtxt(data_2_location, delimiter = ',', skiprows = x0+2, max_rows = h, usecols = tuple(range(y0+1, y0+w+1)))
#     data = data + data_2 
# =============================================================================
    
    #data[]
    if transpose:
        data = data.T 

# =============================================================================
#     if crop:
# # =============================================================================
# #         x = int(h/2)
# #         y = int(w/2) 
# #         radius = 400 
# #         data = data[x-radius:x+radius+1, y-radius:y+radius+1] 
# # =============================================================================
#         shape = np.shape(data) 
#         print(shape) 
#         circle_row = (shape[1]-1)/2
#         circle_col = (shape[0]-1)/2 
#     else:
#         circle_row = 512
#         circle_col = 512 
#         radius = 400 
# =============================================================================
      
    if log_plot:
        plt.imshow(np.log(np.abs(data)), cmap='inferno', vmin=0) 
        circle = plt.Circle((circle_row,circle_col), radius, color='black', fill=False, lw=3, ls='--')
    else:
        plt.imshow(data, cmap='inferno', vmin=0)
        circle = plt.Circle((circle_row,circle_col), radius, color='white', fill=False, lw=3, ls='--')
    plt.colorbar() 
    plt.title(title)
    
    # Make ticks and tick labels 
    ax = plt.gca() 
    ax.set_xticks([circle_col, circle_col+radius*1/lim, circle_col-radius*1/lim, circle_col-radius*0.5/lim, circle_col+radius*0.5/lim])
    ax.set_xticklabels([0, 1, -1, -0.5, 0.5]) 
    ax.set_yticks([circle_row, circle_row+radius*1/lim, circle_row-radius*1/lim, circle_row-radius*0.5/lim, circle_row+radius*0.5/lim])
    ax.set_yticklabels([0, -1, 1, 0.5, -0.5])
    plt.xlabel('$k_x/k_0$') 
    plt.ylabel('$k_y/k_0$')
    
    if outline_NA:
        ax.add_patch(circle)
        plt.text(780,900,'NA = 1.3', fontsize = 10, bbox=dict(boxstyle='round', facecolor='white'))
    
    if save:
        plt.savefig(data_file[:-4]+'.png', dpi=300, bbox_inches='tight')
        #plt.savefig('LED=8mW-405sp-100x_sample(520nm-ref)_450lp-dualpol-1sec_bfp.png', dpi=300, bbox_inches='tight')
    
    plt.show() 


def plot_bfp_off_center(data_file, title, callibration_bfp_file, crop=True, log_plot = False, outline_NA = False, save = False):
    # Load and normalize the data 
    bfp  = np.loadtxt(callibration_bfp_file, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1,1025)))
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1,1025)))
    data = data/np.max(bfp) 
    
    # Locate the center of the emission circle (regardless of diameter) 
    im = np.array(bfp*255, dtype=np.uint8) # Convert to cv2 grayscale 
    ret, thresh = cv2.threshold(im, 127, 255, 0) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    #print(np.sort(areas)) 
    (col,row),radius = cv2.minEnclosingCircle(contours[areas.index(np.sort(areas)[-1])])
    print(col)
    print(row)
    print(radius) 
    #radius = 400 # Leave this at 400  
    
    if crop: 
        col = int(np.round(col))
        row = int(np.round(row)) 
        radius = int(np.round(radius)) 
        data = data[col-radius:col+radius+1, row-radius:row+radius+1] 
        shape = np.shape(data) 
        #print(shape) 
        col = int(np.round((shape[1]-1)/2))
        row = int(np.round((shape[0]-1)/2)) 
    print(col)
    print(row)
    print(radius) 
    
    if log_plot:
        plt.imshow(np.log(np.abs(data)), cmap='inferno') 
        circle = plt.Circle((row, col), radius, color='black', fill=False, lw=3, ls='--')
    else:
        plt.imshow(data, cmap='inferno')
        circle = plt.Circle((row, col), radius, color='white', fill=False, lw=3, ls='--')
    plt.colorbar() 
    plt.title(title)
    
    # Make ticks and tick labels 
    ax = plt.gca() 
    ax.set_xticks([row, row+radius*1/1.3, row-radius*1/1.3, row+radius*0.5/1.3, row-radius*0.5/1.3])
    ax.set_xticklabels([0, 1, -1, 0.5,-0.5]) 
    ax.set_yticks([col, col+radius*1/1.3, col-radius*1/1.3, col+radius*0.5/1.3, col-radius*0.5/1.3])
    ax.set_yticklabels([0, -1, +1, -0.5, +0.5])
    plt.xlabel('$k_x/k_0$') 
    plt.ylabel('$k_y/k_0$')
    
    if outline_NA: 
        ax.add_patch(circle)
        plt.text(780,900,'NA = 1.3', fontsize = 10, bbox=dict(boxstyle='round', facecolor='white'))
    
    if save:
        plt.savefig(data_file[:-4] + '.png', dpi=300, bbox_inches='tight')  
    
    plt.show() 
 
def plot_Ek_off_center(data_file, title, callibration_bfp_file, crop=True, save=False): 
    # Load and normalize the data 
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
    lam  = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1, usecols = tuple(range(1,1025)))
    bfp  = np.loadtxt(callibration_bfp_file, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1,1025)))
    data = data/np.max(bfp) 
    
    # Locate the center of k (using callibration_bfp) 
    im = np.array(bfp*255, dtype=np.uint8) # Convert to cv2 grayscale 
    ret, thresh = cv2.threshold(im, 127, 255, 0) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    (row,col),radius = cv2.minEnclosingCircle(contours[areas.index(np.sort(areas)[-1])])
    
    if crop: 
        row = int(row)
        col = int(col)
        radius = 400 
        data = data[row-radius:row+radius+1, col-radius:col+radius+1] 
        lam = lam[col-radius:col+radius+1]
        shape = np.shape(data) 
        #print(shape) 
        #kx0 = (shape[1]-1)/2
        ky0 = (shape[0]-1)/2 
    
    plt.imshow(data, cmap='inferno')
    plt.colorbar() 
    plt.title(title)
    
    # Make ticks and tick labels 
    wavelength_labels = [] 
    wavelength_indices = [] 
    for i in range(len(lam)): 
        w = int(lam[i]) 
        if (w%20 == 0) and (np.isin(str(w), wavelength_labels)== False): 
            wavelength_labels.append(str(int(w)))  
            wavelength_indices.append(i) 
    #print(wavelength_labels)  
    ax = plt.gca() 
    ax.set_xticks(wavelength_indices)
    ax.set_xticklabels(wavelength_labels)  
    ax.set_yticks([ky0, ky0+radius*1/1.3, ky0-radius*1/1.3, ky0+radius*0.5/1.3, ky0-radius*0.5/1.3])
    ax.set_yticklabels([0, -1, 1, 0.5, -0.5])
    plt.xlabel('Wavelength (nm)') 
    plt.ylabel('$k_y/k_0$')
    
    if save:
        plt.savefig(data_file[:-4] + '.png', dpi=300, bbox_inches='tight')  
    
    plt.show() 
    
def plot_Ek_centered(data_file, NA, title, crop_k=True, crop_lambda=False, save=False, log_plot = False, outline_NA=False): 
    # Load and normalize the data 
    match NA:
        case '1.3':
            (x0, y0), (w, h) = ((112,112),(801,801))
            lim = 1.3
        case '1.25':
            (x0, y0), (w, h) = ((435,435),(160,160)) 
            lim = 1.25
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = x0+3, max_rows = h, usecols = tuple(range(1, 1024)))
    lam  = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1, usecols = tuple(range(1,1025)))
    radius = (h-1)/2
    shape = np.shape(data) 
    print(shape) 
    
# =============================================================================
#     # Reverse the order, if needed 
#     data = data[:,::-1]
#     lam= lam[::-1]
# =============================================================================
    
# =============================================================================
#     circle_row = (shape[1]-1)/2
#     circle_col = (shape[0]-1)/2 
# =============================================================================
    
# =============================================================================
#     data  = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
#     lam  = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1, usecols = tuple(range(1,1025)))
# =============================================================================
    #print(np.min(data)) 
    #print(np.argwhere(data==np.min(data))) 
    
# =============================================================================
#     row = 512
#     col = 512 
#     radius = 400 
#     if crop_k: 
#         data = data[row-radius:row+radius+1, :] 
#         shape = np.shape(data) 
#         #print(shape) 
#         row = (shape[0]-1)/2 
# =============================================================================
        
       
    # Make ticks and tick labels 
    wavelength_labels = [] 
    wavelength_indices = [] 
    if crop_lambda != False: # crop_lambda is either 'False' or a tuple of values representing the lower and upper limits of the crop
        # First crop the data
        lam_min_index = np.argmin(np.abs(lam - crop_lambda[1]))
        lam_max_index = np.argmin(np.abs(lam - crop_lambda[0]))
        lam = lam[lam_min_index : lam_max_index + 1]
        data = data[:, lam_min_index : lam_max_index + 1]  
        # Then make the labels 
        for i in range(len(lam)):
            w = int(lam[i])
            if (w%20 == 0) and (np.isin(str(w), wavelength_labels) == False):
                wavelength_labels.append(str(int(w)))  
                wavelength_indices.append(i) 
    else:
        for i in range(len(lam)): 
            w = int(lam[i]) 
            if (w%20 == 0) and (np.isin(str(w), wavelength_labels)== False): 
                wavelength_labels.append(str(int(w)))  
                wavelength_indices.append(i) 
    
# =============================================================================
#     # Truncate max of data so we can see the background counts 
#     for i in range(len(data[0:,0])):
#         for j in range(len(data[0,:])):
#             if data[i,j] > 100:
#                 data[i,j] = 100
# =============================================================================
            
    if log_plot:
        plt.imshow(np.log(np.abs(data)), cmap='inferno', vmin=0, aspect='auto') 
    else:
        plt.imshow(data, cmap='inferno', vmin=0, aspect='auto')

    plt.colorbar() 
    plt.title(title)
        
    ax = plt.gca() 
    ax.set_xticks(wavelength_indices)
    ax.set_xticklabels(wavelength_labels)  
    ax.set_yticks([radius, radius*(1+1/lim), radius*(1-1/lim), radius*(1+0.5/lim), radius*(1-0.5/lim)])
    ax.set_yticklabels([0, -1, 1, -0.5, 0.5])
    #ax.set_yticklabels([])
    plt.xlabel('Wavelength (nm)') 
    plt.ylabel('$k_y/k_0$')
        
    if save:
        plt.savefig(data_file[:-4] + '.png', dpi=300, bbox_inches='tight')  
    
    plt.show() 
    
def correction_075NA(data_file_1, data_file_2):
    data1  = np.loadtxt(data_file_1, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1,1025)))
    
    #plt.imshow(data) 
    ppol1 = np.array([])
    spol1 = np.array([])
    for i in range(np.size(data1[:,0])):
        ppol1 = np.append(ppol1, (data1[i,i]+data1[1023-i,1023-i])/2)
        spol1 = np.append(spol1, (data1[i, 1023-i]+data1[1023-i,i])/2)
    
    #plt.plot(ppol1) 
    #plt.plot(spol1) 
    
    dualpol1 = (ppol1 + spol1)/2
    plt.plot(dualpol1, 'red') 
    
    # Repeat for data #2 
    data2  = np.loadtxt(data_file_2, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1,1025)))
    
    #plt.imshow(data) 
    ppol2 = np.array([])
    spol2 = np.array([])
    for i in range(np.size(data2[:,0])):
        ppol2 = np.append(ppol2, (data2[i,i]+data2[1023-i,1023-i])/2)
        spol2 = np.append(spol2, (data2[i, 1023-i]+data2[1023-i,i])/2)
    
    #plt.plot(ppol2) 
    #plt.plot(spol2) 
    
    dualpol2 = (ppol2 + spol2)/2
    plt.plot(dualpol2, 'blue') 
    
    correction = (dualpol1 + dualpol2) / 2
    print(correction) 
    correction = correction[100:-100] 
    print(np.shape(correction))
    
    plt.plot(correction) 
    
    plt.show() 
    plt.plot(ppol1[100:-100] / correction)
    plt.show() 
    #np.savetxt("correction.csv", correction, delimiter=",")
    return np.round(correction) 

def plot_spectrum(data_file, title, rows_to_plot, crop_k=True, crop_lambda=False, save=False):
    # Data file (string): directory location of the csv file to be plotted
    # Title (string): Title to display on the plot figure
    # rows_to_plot (tuple): First and last rows in the range over which to sum (inclusive) before plotting 
    
    # Load and normalize the data 
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
    lam  = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1, usecols = tuple(range(1,1025)))
    #print(np.min(data)) 
    #print(np.argwhere(data==np.min(data))) 
    
    
    data = np.sum(data[rows_to_plot[0]:rows_to_plot[1]+1, :], axis=0)
    #data = data/np.max(data) 
    
    # Make ticks and tick labels 
    wavelength_labels = [] 
    wavelength_indices = [] 
    
    # Crop lambda 
    if crop_lambda != False: # crop_lambda is either 'False' or a tuple of values representing the lower and upper limits of the crop
        # First crop the data
        lam_min_index = np.argmin(np.abs(lam - crop_lambda[0]))
        lam_max_index = np.argmin(np.abs(lam - crop_lambda[1]))
        lam = lam[lam_min_index : lam_max_index + 1]
        data = data[lam_min_index : lam_max_index + 1]  
        # Then make the labels 
        for i in range(len(lam)):
            w = int(lam[i])
            if (w%20 == 0) and (np.isin(str(w), wavelength_labels) == False):
                wavelength_labels.append(str(int(w)))  
                wavelength_indices.append(i) 
    else:
        for i in range(len(lam)): 
            w = int(lam[i]) 
            if (w%20 == 0) and (np.isin(str(w), wavelength_labels)== False): 
                wavelength_labels.append(str(int(w)))  
                wavelength_indices.append(i)  
    
    
# =============================================================================
#     for i in range(len(lam)): 
#         w = int(lam[i]) 
#         if (w%20 == 0) and (np.isin(str(w), wavelength_labels)==False): 
#             wavelength_labels.append(str(int(w)))
#             wavelength_indices.append(i) 
# =============================================================================
    
    plt.plot(data)
    plt.title(title)
        
    ax = plt.gca() 
    ax.set_xticks(wavelength_indices)
    ax.set_xticklabels(wavelength_labels)  
    #ax.set_yticklabels([])
    plt.xlabel('Wavelength (nm)') 
    plt.ylabel('Counts (a.u.)')
    
    if save:
        plt.savefig(data_file[:-8]+'spectrum.png', dpi=300, bbox_inches='tight')
    
    plt.show() 
    
    return np.array([lam, data])   
    
def get_max(data_file):
    # Load and normalize the data 
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
    return np.max(data)  

def get_avg(data_file, background_file, dimensions, bkgrnd_choice):
    # Load and normalize the data 
    # 'dimensions' is a tuple of tuples, ((x0,y0),(x1,y1))
        # where (x0,y0) is the top left corner of the region of interest
        # and (x1,y1) is the bottom right corner of the RoI 
        # w and h are the width and height of the ROI, respectively 
    (x0, y0), (x1, y1) = dimensions 
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    data  = np.loadtxt(data_file, delimiter = ',', skiprows = 2+y0, max_rows = h, usecols = tuple(range(x0, x1)))
    if bkgrnd_choice == 'sapphire':
        bkgrnd = np.loadtxt(background_file, delimiter = ',', skiprows = 2+y0, max_rows = h, usecols = tuple(range(x0, x1)))
        return np.round( (np.sum(np.sum(data)) - np.sum(np.sum(bkgrnd))) / (w*h) ) # returns average counts minus average background 
    elif bkgrnd_choice == 'k-mirror':
        bkgrnd = np.loadtxt(data_file, delimiter = ',', skiprows = 2+(1024-y0+1), max_rows = h, usecols = tuple(range(x0, x1)))
        return np.round( (np.sum(np.sum(data)) - np.sum(np.sum(bkgrnd))) / (w*h) ) # returns average counts minus average background 
    elif bkgrnd_choice == 'full_kspace':
        bkgrnd = np.loadtxt(data_file, delimiter = ',', skiprows = 2, max_rows = 1024, usecols = tuple(range(1, 1024)))
        return np.round( np.sum(np.sum(data))/(w*h) - np.sum(np.sum(bkgrnd))/(1024**2))
    elif bkgrnd_choice == 'None':
        return np.round(np.sum(np.sum(data))/(w*h)) 

#data_folder = '/home/wkmills/Dropbox/research/measurements/directional_meta-LEDs/2025-04-14/'
#data_name = 'B9_100um_bfp'
data_location = os.getcwd() 
#bfp_file = data_location + 'bfp.csv' 
data_file = data_location + '/LED=3,5mW-405sp-100x_sample(5,9)_450lp-spol-5sec_E(k).csv' 
'/NA1,25-objective_kspace_+-1k0.csv'

# =============================================================================
# D = get_avg(data_file, None, ((512-13,512-13),(512+13,512+13)), 'None') / get_avg(data_file, None, ((112,112),(801,801)), 'None')
# D *= np.pi/4 # To account for all the intensity=0 values outside NA=1.3
# print('D = ' + str(D)) 
# =============================================================================

#plot_Ek_off_center(data_file, 'Grating at 532 nm', bfp_file) 
#plot_bfp_off_center(data_file, 'No slit, with laser', bfp_file, outline_NA=True, crop=True) 
#plot_bfp_centered(data_file, '1.3', 'Unpolarized PL from metallized sample \n 1-sec integration', save=False, transpose=True )#' \n D = {:.2f}'.format(D), save=False, transpose=True)
#plot_Ek_centered(data_file, '1.3', 'PL from 520 nm-reference \n Pump: 8 mW LED, < 405 nm \n Output: s-pol, 1 sec integration')#, crop_lambda=(521, 440), save=False)
#c = correction_075NA(data_location + '/40x-objective_bfp_BL-largest-position.csv', data_location + '/40x-objective_bfp_BL-smallest-position.csv')
'1.8-μm GaN on DSP' 'quick response' '(0, -0.5k_0)'
spectrum = plot_spectrum(data_file, 'PL from meta-LED; device (5,9) \n Pump: 3.5 mW LED, < 405 nm \n Output: s-pol, 5 sec integration', (200,800), crop_lambda=(450,545), save=True)

# =============================================================================
# analyzer_str = np.array(['spol+90deg', 'pol+80deg', 'pol+70deg', 
#                      'pol+60deg', 'pol+50deg', 'pol+40deg',
#                      'pol+30deg', 'pol+20deg', 'pol+10deg', 
#                      'ppol00deg', 
#                      'pol-10deg', 'pol-20deg', 'pol-30deg', 
#                      'pol-40deg', 'pol-50deg', 'pol-60deg',
#                      'pol-70deg', 'pol-80deg', 'spol-90deg'])
# analyzer_int = np.arange(90, 280, 10)
# =============================================================================
# =============================================================================
# analyzer_str = np.append('ppol000deg', np.array(['pol{:03d}deg'.format(a) 
#                                                  for a in np.arange(10,370,10)]),) 
# analyzer_int = np.arange(0, 370, 10) #[int(a) for a in analyzer_str] 
# spump = np.array([])
# ppump = np.array([])
# 
# m = 1 # Which lobe to plot in the switch statement below 
# for key in analyzer_str:
#     # Loop over analyzer angles and add get_avg() to the arrays created above 
#     match m:
#         case -1:
#             #s_lobe_limits = ((480, 835), (554, 865)) # s-pump, m = -1 lobe
#             #p_lobe_limits = ((480, 835), (554, 865)) # p-pump, m = -1 lobe
#             print('Last I checked, there was no m=-1 lobe to plot')
#         case 0:
#             s_lobe_limits = ((507, 497), (521, 504)) # s-pump, m = 0 lobe
#             p_lobe_limits = ((515, 554), (529, 568)) # p-pump, m = 0 lobe
#         case 1:
#             s_lobe_limits = ((504, 184), (528, 194)) # s-pump, m = +1 lobe
#             p_lobe_limits = ((515, 247), (531, 263)) # p-pump, m = +1 lobe
#     #lobe_limits = ((484, 659), (575, 676)) # m = 0 lobe at 0.5*k0 
#     spump_file = os.getcwd() + '/75mW-1080nm-spol-(0,-0,15k0)_metasurface_' + key + '_1sec.csv'
#     spump_bkgrnd = os.getcwd() + '/75mW-1080nm-spol-(0,-0,15k0)_metasurface_' + key + '_1sec.csv'
#     spump = np.append(spump, get_avg(spump_file, spump_bkgrnd, s_lobe_limits , 'k-mirror')) 
#     ppump_file = os.getcwd() + '/75mW-1080nm-ppol-(0,-0,15k0)_metasurface_' + key + '_1sec.csv'
#     ppump_bkgrnd = os.getcwd() + '/75mW-1080nm-ppol-(0,-0,15k0)_metasurface_' + key + '_1sec.csv'
#     ppump = np.append(ppump, get_avg(ppump_file, ppump_bkgrnd, p_lobe_limits , 'k-mirror')) 
#     
# # =============================================================================
# #     if ('090' in key) or ('180' in key) or ('270' in key) or ('000' in key): 
# #         save = True 
# #         plot_Ek_centered(spump_file, '1.3', 'SHG from metasurface \n pump at 1080 nm, $k_\parallel / k_0 = 0.15$ \n s-pol input, ' + key + ' output \n 1 sec integration', crop_lambda=(520, 560), save = save)
# #         #plot_Ek_centered(spump_bkgrnd, '1.3', 'Background from sapphire \n pump at 1080 nm, $k_\parallel / k_0 = 0.15$ \n s-pol input, ' + key + ' output \n 1 sec integration', crop_lambda=(520, 560), save = save)
# #         plot_Ek_centered(ppump_file, '1.3', 'SHG from metasurface \n pump at 1080 nm, $k_\parallel / k_0 = 0.15$ \n p-pol input, ' + key + ' output \n 1 sec integration', crop_lambda=(520, 560), save = save)
# #         #plot_Ek_centered(ppump_bkgrnd, '1.3', 'Background from sapphire \n pump at 1080 nm, $k_\parallel / k_0 = 0.15$ \n p-pol input, ' + key + ' output \n 1 sec integration', crop_lambda=(520, 560), save = save)
# # =============================================================================
# =============================================================================

# =============================================================================
# # Plot with 'x' and 'o' markers (measurements) 
# plt.plot(analyzer_int, np.abs(spump), marker='x', linestyle='', label='s-pol pump')
# plt.plot(analyzer_int, np.abs(ppump), marker='o', linestyle='', label= 'p-pol pump')
# # =============================================================================
# # # Plot with lines (theory)
# # plt.plot(analyzer_int, spump, label='s-pol pump')
# # plt.plot(analyzer_int, ppump, label= 'p-pol pump')
# # =============================================================================
# #plt.plot(analyzer_int, 2750*np.sin(np.deg2rad(analyzer_int[9:19]))**2, color = 'blue')
# #plt.plot(analyzer_int, 4000*np.cos(np.deg2rad(analyzer_int[9:19]))**2, color='orange')
# plt.xlabel('Analyzer angle \n(0 = p-pol, $\pm$90 = s-pol)')
# plt.xticks(np.arange(0,361,45))
# plt.title('SHG from metasurface, $m=$' + str(m) + ' mode \n Pump @ 75 mW, 1080 nm, $k_\parallel / k_0 = -0.15$ \n 1 sec integration \n Using k-mirrored data as background')
# plt.legend() 
# #plt.ylim(bottom=0)
# plt.show() 
# =============================================================================

# =============================================================================
# for paths, dirs, files in os.walk(os.getcwd()):
#     for file in files:
#         if (file.startswith('ppol') and ('ppol' in file)): 
#             print(file + '\nsum = ' + str(get_avg(data_location + '/' + file, (265, 90))))
# =============================================================================
        
# =============================================================================
# # Plot power dependence 
# power_range = np.array([0, 9, 35, 75, 124])
# SHG = np.array([]) 
# lobe_limits = ((515, 554), (529, 568))
# for p in power_range:
#     data_file = data_location + '/' + str(p) + 'mW-1080nm-spol-(0,-0,15k0)_thinfilm_ppol_10sec.csv'
#     SHG = np.append(SHG, get_avg(data_file, data_file, lobe_limits, 'k-mirror'))
# 
# # Fit and plot 
# a = np.polynomial.polynomial.Polynomial.fit(power_range, SHG, deg = [2], window=[0,124]).convert().coef[2]
# plt.plot(power_range, SHG, marker='x', linestyle='')
# plt.plot(power_range, a*power_range**2, label='$x^2$ fit', color='#e69f00')
# plt.legend() 
# plt.xlabel('Pump power (mW)')
# plt.ylabel('SHG counts')
# plt.title('Power dependence of SHG from thin film')
# plt.show() 
# =============================================================================

# =============================================================================
# # Plot k dependence 
# k_range = np.arange(0.05, 1.25, 0.1)
# k_range_str = np.append( np.array(['-0,{:02}'.format(int(k*100)) for k in k_range[:-2]]), ['-1,05', '-1,15'])
# SHG_spump = np.array([])
# SHG_ppump = np.array([])
# # =============================================================================
# # lobe_limits = np.array([((0,0),(0,0)), #k = -0.05
# #                         ((0,0),(0,0)), #k = -0.15 
# #                         ((0,0),(0,0)), #k = -0.25
# #                         ((0,0),(0,0)), #k = -0.35
# #                         ((0,0),(0,0))  #k = -0.95
# #                         ])
# # =============================================================================
# for ki in range(len(k_range_str)):
#     k_str = k_range_str[ki] 
#     print(k_str) 
#     if ki <= 9:
#         p_data_file = data_location + '/75mW-1080nm-ppol-(0,' + k_str + 'k0)_thinfilm_ppol_1sec.csv'
#         s_data_file = data_location + '/75mW-1080nm-spol-(0,' + k_str + 'k0)_thinfilm_ppol_1sec.csv'
#         SHG_ppump = np.append(SHG_ppump, get_max(p_data_file))
#         SHG_spump = np.append(SHG_spump, get_max(s_data_file)) 
#     else:
#         p_data_file = data_location + '/75mW-1080nm-ppol-(0,' + k_str + 'k0)_thinfilm_ppol_100ms.csv'
#         s_data_file = data_location + '/75mW-1080nm-spol-(0,' + k_str + 'k0)_thinfilm_ppol_100ms.csv'
#         SHG_ppump = np.append(SHG_ppump, 10*get_max(p_data_file))
#         SHG_spump = np.append(SHG_spump, 10*get_max(s_data_file)) 
#     
# plt.plot(k_range, SHG_spump/1e3, marker='x', linestyle='', label='s-pol')
# plt.plot(k_range, SHG_ppump/1e3, marker='x', linestyle='', label='p-pol')
# plt.xlabel('$k_\parallel / k_0$')
# plt.title('p-polarized SHG from thin film \n pump @ 75 mW, 1080 nm')
# plt.ylabel('Counts (a.u.)')
# plt.legend() 
# #plt.savefig('k-dependence.png', dpi=300, bbox_inches='tight')
# plt.show()     
# =============================================================================

# =============================================================================
# # Plot k-dependence as vertical slice of heatmap, for p/p, s/p, s/s, and p/s 
# #k_string = '7' # ky/k0 = -0._5 
# pol = 'pp'
# pump_or_SHG = 'pump'
# k_detector = np.linspace(+1.64, -1.64, 1024) 
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
# #pol = {'name' : ['pp', 'sp', 'ss', 'ps']}
# #for pol in ['pp', 'sp', 'ss', 'ps']:
# k_range_string = ['-0,55', '-0,50', '-0,45', '-0,40', '-0,35', '-0,30', '-0,25', '-0,20', '-0,15', '-0,10', '-0,05',
#                   '-0,0', 
#                   '+0,05', '+0,10', '+0,15', '+0,20', '+0,25', '+0,30', '+0,35', '+0,40', '+0,45', '+0,50', '+0,55'] 
# k_range_float = np.arange(-0.55, 0.6, 0.05)
# k_range_string = k_range_string[2:21]
# k_range_float = k_range_float[2:21]
# 
# N_datapoints = len(k_range_float) 
# ax = plt.gca() 
# # Find which column to plot 
# data_file = data_location + '/75mW-1080nm-' + pol[0] + 'pol-(0,-0,0k0)_metasurface_' + pol[1] + 'pol-540nm-5sec.csv'
# data_0 = np.loadtxt(data_file, delimiter = ',', skiprows = int(3+(1024/2)-25), max_rows = 50, usecols = tuple(range(int(1+(1024/2)-25), int((1024/2)+25))) )
# col_index = int((1024/2)-25 + np.unravel_index(np.argmax(data_0, axis=None), data_0.shape)[1])
# for i in range(N_datapoints): 
#     k_string = k_range_string[i] 
#     k = k_range_float[i] 
#     if pump_or_SHG == 'pump':
#         #data_file = data_location + '/75mW-1080nm-' + pol[0] + 'pol-(0,' + k_string + 'k0)_metasurface_' + pol[1] + 'pol-540nm-5sec.csv'
#         data_file = data_location + '/75mW-1080nm-' + pol[0] + 'pol-(0,' + k_string + 'k0)_metasurface_' + pol[1] + 'pol-1080nm-10ms.csv' 
#     elif pump_or_SHG == 'SHG':
#         #data_file = data_location + '/75mW-1080nm-' + pol[0] + 'pol-(0,' + k_string + 'k0)_metasurface_' + pol[1] + 'pol-540nm-5sec.csv'
#         data_file = data_location + '/75mW-1080nm-' + pol[0] + 'pol-(0,' + k_string + 'k0)_metasurface_' + pol[1] + 'pol-540nm-5sec.csv'
#     data = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
#     #label = pol
#     #plot_Ek_centered(data_file, '1.3', k_string)
#     #label = k_string + '$k_0$'
#     if i < ((N_datapoints-1)/2 - 7):
#         plt.plot(k_detector, np.average(data[:,col_index-10:col_index+10],1) - np.average(data[:,258]), marker='v', markersize=3, linestyle='none', color = colors[int(i + 14 - (N_datapoints-1)/2)])
#     elif i < ((N_datapoints-1)/2):
#         plt.plot(k_detector, np.average(data[:,col_index-10:col_index+10],1) - np.average(data[:,258]), marker='^', markersize=3, linestyle='none', color = colors[int(i + 7 - (N_datapoints-1)/2)])
#         #plt.axvline(k, 0.96, 1.0, color = colors[i])
#         #plt.plot(k, 1600, 'o', markersize=3, color=colors[i])
#     elif i == ((N_datapoints-1)/2):
#         plt.plot(k_detector, np.average(data[:,col_index-10:col_index+10],1) - np.average(data[:,258]), marker='o', markersize=3, linestyle='none', color='black')#, zorder=1.5)
#         #plt.axvline(k, 0.96, 1.0, color = 'black')
#         #plt.plot(k, 1600, 'o', markersize=3, color='black')
#     elif i < ((N_datapoints+1)/2 + 7): 
#         plt.plot(k_detector, np.average(data[:,col_index-10:col_index+10],1) - np.average(data[:,258]), marker='+', markersize=4, linestyle='none', color = colors[-int(((i + 7 - (N_datapoints-1)/2)%8)+1)])
#         #plt.axvline(k, 0.96, 1.0, color = colors[-(i%8)-1])
#         #plt.plot(k, 1600, '+', markersize=4, color=colors[-(i%8)-1])
#     else: 
#         plt.plot(k_detector, np.average(data[:,col_index-10:col_index+10],1) - np.average(data[:,258]), marker='x', markersize=4, linestyle='none', color = colors[int((N_datapoints+1)/2 + 13)-i])
# 
# # Add legend
# ymax = plt.gca().get_ylim()[1] 
# plt.plot([0.6, 0.6, -0.6, -0.6, 0.6], [0.94*ymax, 0.90*ymax, 0.90*ymax, 0.94*ymax, 0.94*ymax], 'black', linewidth=0.8)
# for i in range(N_datapoints): 
#     k = k_range_float[i] 
#     if i < ((N_datapoints-1)/2 - 7):
#         plt.plot(k, 0.92*ymax, 'v', markersize=3, color=colors[int(i + 14 - (N_datapoints-1)/2)])
#     elif i < ((N_datapoints-1)/2):
#         plt.plot(k, 0.92*ymax, '^', markersize=3, color=colors[int(i + 7 - (N_datapoints-1)/2)])
#     elif i == ((N_datapoints-1)/2):
#         plt.plot(k, 0.92*ymax, 'o', markersize=3, color='black')
#     elif i < ((N_datapoints+1)/2 + 7): 
#         plt.plot(k, 0.92*ymax, '+', markersize=4, color=colors[-int(((i + 7 - (N_datapoints-1)/2)%8)+1)])
#     else:
#         plt.plot(k, 0.92*ymax, 'x', markersize=3, color=colors[int((N_datapoints+1)/2 + 13)-i])
# 
# if pump_or_SHG == 'pump':
#     plt.title('Pumping Larry\'s dual-pol metasurface \n Pump: 75 mW @ 1080 nm, ' + pol[0] + '/' + pol[1] + ' polarization \n 10-ms integration, reflected $k_{pump}$ marked') 
# elif pump_or_SHG == 'SHG':
#     plt.title('SHG from Larry\'s dual-pol metasurface \n Pump: 75 mW @ 1080 nm, ' + pol[0] + '/' + pol[1] + ' polarization \n 5-sec integration, reflected $k_{pump}$ marked')
# plt.xlim([-1.3, 1.3])
# plt.xlabel('$k_y / k_0$')
# 
# SAVE = False  
# if SAVE:
#     if pump_or_SHG == 'pump':
#         if pol == 'sp' or pol == 'ps':
#             answer = input('Are you sure you want to save? Mixed-polarization pump plots may not be useful. Y or N: ')
#             if answer == 'Y' or answer == 'y':
#                 plt.savefig(pol[0] + '' + pol[1] + '-pol_pump.png', dpi=300, bbox_inches='tight') 
#             else:
#                 sys.exit()  
#         elif pol =='ss' or pol == 'pp':
#             plt.savefig(pol[0] + '' + pol[1] + '-pol_pump.png', dpi=300, bbox_inches='tight') 
#     elif pump_or_SHG == 'SHG':
#         plt.savefig(pol[0] + '' + pol[1] + '-pol_k-dependence.png', dpi=300, bbox_inches='tight') 
# 
# plt.show() 
# =============================================================================

# =============================================================================
# # Plot ratio of primary SHG lobe to diffracted SHG lobe as a function of k_pump
# k_detector = np.linspace(+1.64, -1.64, 1024) # Full extent of the detector; for creating the x axis 
# k_range_string = ['-0,55', '-0,50', 
#                   '-0,45', '-0,40', '-0,35', '-0,30', '-0,25', '-0,20', '-0,15', '-0,10', '-0,05',
#                   '-0,0', 
#                   '+0,05', '+0,10', '+0,15', '+0,20', '+0,25', '+0,30', '+0,35', '+0,40', '+0,45', '+0,50', '+0,55'] 
# k_range_float = np.arange(-0.55, 0.6, 0.05) # np.arange(-0.45, 0.5, 0.05) 
# N_datapoints = len(k_range_float) 
# normalization = 'efficiency' # 'efficiency' = (mediated peak)/(meadiated peak + primary peak) 
#                              # 'none' = (mediated peak)  
# save = False  
# 
# for pol in ['s','p']:
#     peak_ratios = [] 
#     diff_counts = [] 
#     
#     # Find which column to plot 
#     data_file = data_location + '/75mW-1080nm-' + pol + 'pol-(0,-0,0k0)_metasurface_ppol-540nm-5sec.csv'
#     data_0 = np.loadtxt(data_file, delimiter = ',', skiprows = int(3+(1024/2)-25), max_rows = 50, usecols = tuple(range(int(1+(1024/2)-25), int((1024/2)+25))) )
#     col_index = int((1024/2)-25 + np.unravel_index(np.argmax(data_0, axis=None), data_0.shape)[1])
# 
#     for i in range(N_datapoints):
#         k_string = k_range_string[i] 
#         k = k_range_float[i] 
#         data_file = data_location + '/75mW-1080nm-' + pol + 'pol-(0,' + k_string + 'k0)_metasurface_ppol-540nm-5sec.csv'
#         data = np.loadtxt(data_file, delimiter = ',', skiprows = 3, max_rows = 1024, usecols = tuple(range(1,1025)))
#         primary_peak_index = (np.abs(k_detector - k)).argmin()
#         primary_counts = np.abs(np.max(data[primary_peak_index-50 : primary_peak_index+50, col_index]))
#         diff_peaks_index = [(np.abs(k_detector - (k-1))).argmin(), 
#                             (np.abs(k_detector - (k+1))).argmin()] # location of (+k)- and (-k)-mediated lobes 
#         diff_counts = np.append(diff_counts, np.abs(np.max(np.append(data[diff_peaks_index[0]-50 : diff_peaks_index[0]+50, col_index], 
#                                                                      data[diff_peaks_index[1]-50 : diff_peaks_index[1]+50, col_index])))) 
#         peak_ratios = np.append(peak_ratios, np.abs(diff_counts[i]/(primary_counts+diff_counts)[i]))
#     if normalization == 'efficiency':
#         plt.plot(k_range_float, peak_ratios, label=pol+'-pump', marker='o', linestyle='--')
#     elif normalization == 'none':
#         plt.plot(k_range_float, diff_counts, label=pol+'-pump', marker='o', linestyle='--')
# plt.legend()
# plt.xlim([-0.6, +0.6])
# plt.xlabel('$k_\parallel / k_0$')
# if normalization == 'efficiency':
#     plt.ylabel('Diffraction efficiency \n (using peak counts of each lobe)')
#     plt.title('Diffraction efficiency vs pump angle') 
#     if save: plt.savefig('SHG-diffraction-efficiency.png', dpi=300, bbox_inches='tight') 
# elif normalization == 'none':
#     plt.ylabel('Lobe intensity (a.u.)')
#     plt.title('Intensity of \"diffracted\" SHG lobe')
#     if save: plt.savefig('SHG-diffraction-intensity.png', dpi=300, bbox_inches='tight') 
# #plt.savefig('SHG-diffraction-intensity.png', dpi=300, bbox_inches='tight') 
# plt.show() 
# =============================================================================
        


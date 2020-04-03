#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:04:08 2019

@author: aryaa
"""

import math
import numpy as np # using for  array manipulations
import sys
import cv2 #  opening and writing images to the device

# Gaussian smoothing function 7x7
def smoothing(img, r, c):        # for  Smoothing function defined
    temp = np.zeros((gr,gc),dtype = np.float)  # For creating Temporary array , which takes a slice of orignal array for direct matrix multiplication  
    smoothing_output = np.zeros((r,c),dtype = np.float) # New array created for storing smoothed image after matrix multiplication     
    for i in range(r-gr+1):
        for j in range(c-gc+1):
            temp = img[i:(gr+i),j:(gc+j)] # For taking slice of size of gaussian mask in temporary array        
            smoothing_output[(3+i),(3+j)] = np.sum(np.multiply(gaussian_mask, temp))  # For performing convolution by directly multiplying the matrices and taking sum of it, divide by sum of mask and store in appropriate position
    return smoothing_output  # Returning the smoothed image

# Sobel operator 3x3 
def gradient(img, r, c,  s_kernel_x, s_kernel_y):
    gx = np.zeros((r,c),dtype = np.float)  #For  Defining gradient x array 
    gy = np.zeros((r,c),dtype = np.float)  # For Defining gradient y array
    gm = np.zeros((r,c),dtype = np.float)  # For Defining the gradient magnitude array
    temp_gradient = np.zeros((s_kernel_x.shape[0],s_kernel_x.shape[1]),dtype = np.float) #For  Defining temporary array that will store the slice of smoothed image array for direct matrix multiplication    
    for i in range(r-8):
        for j in range(c-8):
            temp_gradient = img[(3+i):(6+i),(3+j):(6+j)] # For storing the slice of smoothed image array in temporary array
            gx[4+i,4+j] = np.sum(np.multiply(temp_gradient, s_kernel_x)) # For applying convolution for gradient x by directly multpilying the slice of matrix with sobel x operator
            gy[4+i,4+j] = np.sum(np.multiply(temp_gradient, s_kernel_y)) # For applying convolution for gradient y by directly multiplying the slice of matrix with sobel y operator
    gm = np.hypot(np.absolute(gx),np.absolute(gy))/np.sqrt(2) # Forming the normalized gradient magnitude array by using np.hypot() which takes under root of sum of squares of normalized gradient x and normalized gradient y and then dividing by square root of 2 for normalization       
    return gx,gy,gm # Returning gradient x, gradient y and normalized gradient magnitude 




# non maxima suppression
def non_max_suppression(gx,gy,gm, r, c):      # Defining the function for Non Maxima Suppression    
    ga = np.zeros((r,c), dtype = np.float)    # Defining gradient angle matrix 
    ga = np.degrees(np.arctan2(gy,gx)) # Calculating gradient angle by using np.arctan2() which takes inverse tan element wise, and then wrapping it in np.degrees which converts the radian angle returned by np.arctan2() to degrees         
    ga = (ga+360)%360 # Did this so that every angle in the gradient angle matrix falls between 0 to 360 range                          
    gnms = np.zeros((r,c), dtype = np.float)  # Defining the matrix for Non Maxima Suppression
    for i in range(4,r-4):
        for j in range(4,c-4):
            if(((ga[i,j]>=337.5) or (ga[i,j]<22.5)) or ((ga[i,j]>=157.5) and (ga[i,j]<202.5))): # If the angle falls in these ranges then it is in sector 0
                if((gm[i,j]>gm[i,j+1]) and (gm[i,j]>gm[i,j-1])):  # Since we know angle is in sector 0 we compare the appropriate values                              
                    gnms[i,j] = gm[i,j]            # If a local maxima we assign the value of magnitude                                              
            if(((ga[i,j]>=22.5) and (ga[i,j]<67.5)) or ((ga[i,j]>=202.5) and (ga[i,j]<247.5))): # If the angle falls in these ranges then it is in sector 1
                if((gm[i,j]>gm[i-1,j+1]) and (gm[i,j]>gm[i+1,j-1])):   # Since we know angle is in sector 1 we compare the appropriate values                         
                    gnms[i,j] = gm[i,j]      # If a local maxima we assign the value of magnitude                                                    
            if(((ga[i,j]>=67.5) and (ga[i,j]<112.5)) or ((ga[i,j]>=247.5) and (ga[i,j]<292.5))): # if angle falls in these ranges then it is in sector 2
                if((gm[i,j]>gm[i-1,j]) and (gm[i,j]>gm[i+1,j])):    # Since we know angle is in sector 2 we compare the appropriate values                            
                    gnms[i,j] = gm[i,j]    # If a local maxima we assign the value of magnitude                                                     
            if(((ga[i,j]>=112.5) and (ga[i,j]<157.5)) or ((ga[i,j]>=292.5) and (ga[i,j]<337.5))): # if angle falls in these ranges then it is in sector 3
                if((gm[i,j]>gm[i-1,j-1]) and (gm[i,j]>gm[i+1,j+1])):  # Since we know angle is in sector 3 we compare the appropriate values                           
                    gnms[i,j] = gm[i,j]             # If a local maxima we assign the value of magnitude                                               
    return gnms, ga

# double thresholding

# ARYA OLD_ DELETE ME
def doubleThresholding(img, T, ga, r, c):
    t1=T   # Defining first threshold value
    t2=2*T # Defining second threshold value
    img_edge = np.zeros((r, c), dtype = np.int)        # to define the binary edge map which contains all edge points
    for i in range(r):                                  
        for j in range(c):
            if img[i,j]<t1:                          # To check whether pixel magnitude is less than first threshold
                img_edge[i,j] = 0                        #  For setting edge pixel magnitude to zero  
            elif img[i,j]>t2:                        # To check whether pixel magnitude is greater than second threshold
                img_edge[i,j] = 255                      # Setting edge pixel magnitude to 255
            elif t1<=img[i,j]<=t2:                                        # To check whether pixel magnitude is between first and second threshold
                i_arr = [i-1,i,i+1]                      # To define the i-array for 8-connected neighbors
                j_arr = [j-1,j,j+1]                      # To define the j-array for 8-connected neighbors
                for i_ in i_arr:                    
                    for j_ in j_arr:
                        try:                             # For Handling corner pixel neighbors
                            if ga[i,j] > 180: #  For Converting the angles in range (180,360] to (-180, 0]) 
                                ga[i,j] -= 360
                            if ga[i_,j_] > 180: # For Converting the abgles in range (180,360] to (-180, 0]) 
                                ga[i_,j_] -= 360
                            if i_!=j_ and img[i_,j_]>t2 and (abs(abs(ga[i,j])-abs(ga[i_,j_]))<=45):          # For Checking whether neighbor's magintude if greater than 2nd threshold and difference between pixel and it's neighbor's angle is less than or equal to 45 degrees
                                img_edge[i,j] = 255      # For Setting edge pixel magnitude to 255
                            else:
                                img_edge[i,j] = 0        # For  Setting edge pixel magnitude to 0
                        except:
                            continue
    return img_edge    





if __name__ == "__main__":
    file = ['Houses-225', 'Zebra-crossing-1', 'airplane256']
    file = file[2]
    img = cv2.imread(file+".bmp",0)
    r,c = img.shape
    
    gaussian_mask = (1/140)*np.array([(1,1,2,2,2,1,1),        # Defining the gaussian mask
                        (1,2,2,4,2,2,1),
                        (2,2,4,8,4,2,2),
                        (2,4,8,16,8,4,2),
                        (2,2,4,8,4,2,2),gradients[1])) # Printing normalized y gradient
    cv2.imwrite(file+"_gradient_magnitude.bmp",gradients[2])  # Printing gradient magnitude
    
    nms, ga = non_max_suppression(gradients[0], gradients[1], gradients[2], r, c) #image after nms 
    cv2.imwrite(file+"_nms.bmp",nms) # Writing the image after non maxima suppression to device
    
    final = doubleThresholding(nms, 8, ga, r, c) # Put the threshold value here
    cv2.imwrite(file+"final2.bmp",final)  # To write the image to device

                        (1,2,2,4,2,2,1),
                        (1,1,2,2,2,1,1)])
    
    gr,gc = gaussian_mask.shape    # Storing the dimensions of the gaussian mask into variables using shape
    s_kernel_x = (1/4)*np.array([(-1,0,1), (-2,0,2), (-1,0,1)]) # Sobel x operator defined
    s_kernel_y = (1/4)*np.array([(1,2,1), (0,0,0), (-1,-2,-1)]) # Sobel y operator defined
    
    smoothed_img = smoothing(img, r, c)  # We get smoothed_img array after smoothing
    cv2.imwrite(file+"_gaussian.bmp",smoothed_img) # Printing smoothed image
    
    gradients = gradient(smoothed_img, r, c, s_kernel_x, s_kernel_y) #this is a list of three images
    cv2.imwrite(file+"_gradient_x.bmp",np.absolute(gradients[0]))  # Printing normalized x gradient
    cv2.imwrite(file+"_gradient_y.bmp",np.absolute(


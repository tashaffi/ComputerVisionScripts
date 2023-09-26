# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:50:08 2023

@author: Kyle Ross

Revised script for calibrating data
Does not contain the process for averaging the bounding boxes

Input is thermal data from a .cts file
"""


import numpy as np
import os
import cv2 as cv

from skimage import filters

import dataset_functions as df

import numpy as np

def create_dissipating_circle_kernel(radius, sigma):
    """
    Creates a 2D Gaussian kernel with a dissipating circle shape.

    Args:
        radius (int): The radius of the circle.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A 2D numpy array representing the kernel.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (radius, radius)

    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            kernel[x, y] = np.exp(-0.5 * (distance / sigma) ** 2)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel

def otsu(roi,classes):
    #Mask 0s to be empty cells to allow better otsu thresholding without 0 being a class
    while(classes>0):  
        try:
            thresholds = filters.threshold_multiotsu(roi,classes)
            break
        except:
            classes = classes-1
        
    otsu = np.digitize(roi, bins=thresholds)
    return otsu, thresholds


def get_thermal(directory, file):
    thermal_file = os.path.join(directory, file)
    thermal_data =  np.load(thermal_file)
    # thermal_time = thermal_data['time']
    return thermal_data


files = []

# Get all the files for which the calibration doesn't work correctly
summer_data = 'Documents/finalDataRepositorySummer2022'
# get my home directory and join summer_data with it
data_dir = os.path.join(os.path.expanduser('~'), summer_data)
non_calibrated_files = list(np.load(os.path.join(data_dir, 'completedCalibration.npy')))
non_calibrated_files = ['JohnCalibration.npy']


# Example usage:
radius = 4  # Adjust the radius as needed
sigma = .5  # Adjust the sigma (spread) as needed
kernel = create_dissipating_circle_kernel(radius, sigma)

# Print or visualize the kernel
print(kernel)


prev_frame = []
diffs = []
skip = 0
for file in non_calibrated_files:
    file_dir = os.path.join(data_dir, file)
    print("file _----------------------------")
    print(file_dir)
    
    thermal = None
    thermal = get_thermal(data_dir, file)
    
    thermal = np.array(df.resample_data(thermal))
    print(thermal.shape)

    if thermal is not None: 
        for frame in thermal:
            key = cv.waitKey(1000)
            print(frame.shape)
            cv.imshow("Thermal Original", cv.resize(frame.astype('uint8'), (320,320)))
        

            detection = np.copy(frame)
            detection[frame<50] = 0
            # Apply adaptive thresholding
            # detection = cv.cvtColor(detection, cv.COLOR_BGR2GRAY)
            print(detection.shape)
            
            # Apply convolution with the dissipating circle kernel
            result = cv.filter2D(detection, -1, kernel)

            # Define a threshold to identify potential burners
            threshold = 10000  # Adjust as needed
            # Find the minimum value in the result image
            
            '''
            Perform convolution using the dissipating heatmap (which mimics a burner)
            then normalize
            Each iteration of the convolution will increase the intensity of the burner regions
            and suppress the other regions. 
            After doing this a few times, the burner regions will be the only ones with a high intensity, hopefully

            '''
            min_value = abs(np.min(result))
            max_value = abs(np.max(result))
            
            # Normalize the result image by dividing by the minimum value
            result = (result - min_value) / (max_value - min_value)
            result[result<.4] = 0
            result = cv.filter2D(result, -1, kernel)
            
            
            min_value = abs(np.min(result))
            max_value = abs(np.max(result))
            result = (result - min_value) / (max_value - min_value)
            
            result[result<.4] = 0
            result = cv.filter2D(result, -1, kernel)
            min_value = abs(np.min(result))
            max_value = abs(np.max(result))
            result = (result - min_value) / (max_value - min_value)
            
            # Apply the mask to the original image
            frame = np.multiply(result, 255)
            cv.imshow("Normalized Result", cv.resize(frame.astype('uint8'), (320,320)))   
            
            # Find locations where the result exceeds the threshold
            #burner_candidates = np.where(result > threshold)
            contours, _ = cv.findContours(frame.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            
            for _, c in enumerate(contours):
                # Place a bounding box around the contour
                x,y,w,h = cv.boundingRect(c)
                print(x, y, w, h)

                # Check if the bounding box is an appropriate size for a burner
                # if w>=3 and h>=3 and w<=12 and h<=12:
                cv.rectangle(frame,(x,y),(x+w,y+h),(128,128, 55),1)
                # coords.append([x,y,w,h])

            cv.imshow("Detection", cv.resize(frame.astype('uint8'), (320,320)))
            
            if key==27: break


cv.destroyAllWindows()
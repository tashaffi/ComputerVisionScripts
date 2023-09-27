# -*- coding: utf-8 -*-
"""
Revised script for calibrating data
Does not contain the process for averaging the bounding boxes

Input is thermal data from a .cts file
"""

# pylint: disable=E1101

import numpy as np
import os
import cv2

import dataset_functions as df
import numpy as np


## Define parameters

# Define the radius and sigma for the dissipating circle kernel
radius = 4
sigma = .5

# Threshold values
temp_threshold = 50
normalization_threshold = .4

# Convolution iterations
iterations = 2


def stackImages(scale, imgArray, captions=None):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # Resize the image
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                
                # Add captions to the images
                cv2.putText(imgArray[x][y], captions[x][y], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Convert grayscale images to BGR
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            # Resize the image
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            
            # Add captions to the images
            cv2.putText(imgArray[x], captions[x], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convert grayscale images to BGR
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray)
        ver = hor
    
    return ver


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


def get_thermal(directory, file):
    thermal_file = os.path.join(directory, file)
    thermal_data = np.load(thermal_file)
    # thermal_time = thermal_data['time']
    return thermal_data


# Get all the files for which the calibration doesn't work correctly
summer_data = 'Documents/finalDataRepositorySummer2022'
# get my home directory and join summer_data with it
data_dir = os.path.join(os.path.expanduser('~'), summer_data)
non_calibrated_files = list(
    np.load(os.path.join(data_dir, 'completedCalibration.npy')))
non_calibrated_files = ['JohnCalibration.npy']


kernel = create_dissipating_circle_kernel(radius, sigma)


def create_mask(normalization_threshold, kernel, frame, iterations=1):
    '''
            Perform convolution using the dissipating heatmap (which mimics a burner), then normalize.
            Each iteration of the convolution will increase the intensity of the burner regions
            and suppress the other regions. 
            After doing this a few times, the burner regions will be the only ones with a high intensity, hopefully.

            Args:
                normalization_threshold (float): The threshold for the normalized convolution result.
                kernel (numpy.ndarray): The kernel to use for the convolution.
                frame (numpy.ndarray): The frame to convolve.
                iterations (int): The number of iterations to perform.
    '''

    colvolved_frame = frame.copy()
    for i in range(iterations):
        colvolved_frame = cv2.filter2D(colvolved_frame, -1, kernel)
        min_value = abs(np.min(colvolved_frame))
        max_value = abs(np.max(colvolved_frame))
        colvolved_frame = (colvolved_frame - min_value) / (max_value - min_value) # Normalize the convolution result
        colvolved_frame[colvolved_frame < normalization_threshold] = 0

    return colvolved_frame


for file in non_calibrated_files:
    file_dir = os.path.join(data_dir, file)

    thermal = None
    thermal = get_thermal(data_dir, file)
    thermal = np.array(df.resample_data(thermal))
    print(thermal.shape) #check how many frames the thermal image has

    if thermal is not None:
        for i, frame in enumerate(thermal):
            key = cv2.waitKey(1000)
            print(f"Frame: {frame.shape}")

            original_frame = cv2.resize(np.copy(frame).astype('uint8'), (320, 320))
            frame[frame < temp_threshold] = 0 #All regions colder than the threshold should be black in the image
            # detection = cv2.cv2tColor(detection, cv2.COLOR_BGR2GRAY)

            # Apply convolution with the dissipating circle kernel to create a mask
            colvolved_frame = create_mask(normalization_threshold, kernel, frame, iterations=iterations)
            frame = np.multiply(colvolved_frame, 255) # Apply the mask to the original image
            frame[frame < temp_threshold] = 0 #All regions colder than the threshold should be black in the image

            #dilate the edges
            frame = frame.astype(np.uint8)
            frame = cv2.Canny(frame, 100, 180)
            dilation_kernel = np.ones((1, 1), np.uint8)
            frame = cv2.dilate(frame, dilation_kernel, iterations=iterations)

            processed_frame = cv2.resize(np.copy(frame).astype('uint8'), (320, 320))

            contours, _ = cv2.findContours(frame.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for _, c in enumerate(contours):
                x, y, w, h = cv2.boundingRect(c)

                # Check if the bounding box is an appropriate size for a burner
                # if w>=3 and h>=3 and w<=12 and h<=12:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 55), 1)

            images_to_show = [original_frame, processed_frame, cv2.resize(frame.astype('uint8'), (320, 320))]
            captions = [f"Original Frame: {i}", 'Processed Frame', 'Bounding Boxes']
            imgStack = stackImages(1, [images_to_show], [captions])
            cv2.imshow("Visualize Frames",imgStack)

            if key == 27:
                break


cv2.destroyAllWindows()

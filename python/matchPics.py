import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
    # I1, I2 : Images to match
    
    # Convert Images to GrayScale
    if len(I1.shape) == 3:
        img1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    else:
        img1 = I1
        
    if len(I2.shape) == 3:
        img2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    else:
        img2 = I2
    
    # Detect Features in Both Images
    # Using default sigma=0.15 from helper function
    locs1 = corner_detection(img1, sigma=0.15)
    locs2 = corner_detection(img2, sigma=0.15)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(img1, locs1)
    desc2, locs2 = computeBrief(img2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio=0.75)
    
    # Swap x,y coordinates to match expected format (y,x) -> (x,y)
    locs1 = locs1[:, [1, 0]]
    locs2 = locs2[:, [1, 0]]

    return matches, locs1, locs2
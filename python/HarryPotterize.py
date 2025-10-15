import numpy as np
import cv2
import skimage.io 
import skimage.color

# Import necessary functions
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH

# Write script for Q3.9

# Step 1: Read images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

print(f"CV cover size: {cv_cover.shape}")
print(f"CV desk size: {cv_desk.shape}")
print(f"HP cover size: {hp_cover.shape}")

# Step 2: Match features between cv_desk and cv_cover
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
print(f"Found {len(matches)} matches")

# Step 3: Compute homography with RANSAC
bestH2to1, inliers = computeH_ransac(
    locs1[matches[:, 0]],  # x1: points in cv_desk (destination)
    locs2[matches[:, 1]]   # x2: points in cv_cover (source)
)
print(f"RANSAC found {int(np.sum(inliers))} inliers out of {len(matches)} matches")

# Step 4: Resize HP cover to match CV cover dimensions
cv_cover_h, cv_cover_w = cv_cover.shape[:2]
hp_cover_resized = cv2.resize(hp_cover, (cv_cover_w, cv_cover_h))
print(f"Resized HP cover to: {hp_cover_resized.shape}")

# Step 5: Create composite image
composite_img = compositeH(bestH2to1, hp_cover_resized, cv_desk)

# Step 6: Save and display results
cv2.imwrite('HarryPotter_Desk.jpg', composite_img)
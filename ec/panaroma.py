import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from matchPics import matchPics
from planarH import computeH_ransac

# Write script for Q4.2x - Panorama Stitching

# Load images
img_left = cv2.imread('../data/pano_left.jpg')
img_center = cv2.imread('../data/pano_center.jpg')
img_right = cv2.imread('../data/pano_right.jpg')

# Match and compute homographies
matches_CL, locs_C1, locs_L = matchPics(img_center, img_left)
H_CL, _ = computeH_ransac(locs_C1[matches_CL[:, 0]], locs_L[matches_CL[:, 1]])

matches_CR, locs_C2, locs_R = matchPics(img_center, img_right)
H_CR, _ = computeH_ransac(locs_C2[matches_CR[:, 0]], locs_R[matches_CR[:, 1]])

# Calculate panorama size
h, w = img_center.shape[:2]
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

# Project corners through homographies
corners_h = np.hstack([corners, np.ones((4, 1))])
corners_L = (H_CL @ corners_h.T).T
corners_L = corners_L[:, :2] / corners_L[:, 2:3]
corners_R = (H_CR @ corners_h.T).T
corners_R = corners_R[:, :2] / corners_R[:, 2:3]

# Find bounding box
min_x = min(0, corners_L[:, 0].min(), corners_R[:, 0].min())
max_x = max(w, corners_L[:, 0].max(), corners_R[:, 0].max())
min_y = min(0, corners_L[:, 1].min(), corners_R[:, 1].min())
max_y = max(h, corners_L[:, 1].max(), corners_R[:, 1].max())

pano_w = int(np.ceil(max_x - min_x))
pano_h = int(np.ceil(max_y - min_y))

# Translation matrix
T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

# Warp images
warped_left = cv2.warpPerspective(img_left, T, (pano_w, pano_h))
warped_center = cv2.warpPerspective(img_center, T, (pano_w, pano_h))
warped_right = cv2.warpPerspective(img_right, T @ H_CR, (pano_w, pano_h))

# Blend by averaging
panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
weights = np.zeros((pano_h, pano_w), dtype=np.float32)

for img in [warped_left, warped_center, warped_right]:
    mask = (img.sum(axis=2) > 0).astype(np.float32)
    panorama += img.astype(np.float32) * mask[:, :, np.newaxis]
    weights += mask

weights[weights == 0] = 1
panorama = (panorama / weights[:, :, np.newaxis]).astype(np.uint8)

# Save result
cv2.imwrite('panaroma.jpg', panorama)
print(f"Panorama saved: {pano_w}x{pano_h}")
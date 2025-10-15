import numpy as np
import cv2
import sys
sys.path.append('../python')

from matchPics import matchPics
from planarH import computeH_ransac

# Write script for Q4.2x - Panorama Stitching

# Load images
img_left = cv2.imread('../data/pano_left.jpg')
img_right = cv2.imread('../data/pano_right.jpg')

# Match left to right
matches, locs_L, locs_R = matchPics(img_left, img_right)
H_LR, _ = computeH_ransac(locs_L[matches[:, 0]], locs_R[matches[:, 1]])

# Calculate panorama size
h_left, w_left = img_left.shape[:2]
h_right, w_right = img_right.shape[:2]

# Project right corners to left image space
corners_right = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32)
corners_h = np.hstack([corners_right, np.ones((4, 1))])
corners_proj = (H_LR @ corners_h.T).T
corners_proj = corners_proj[:, :2] / corners_proj[:, 2:3]

# Find bounding box
min_x = min(0, corners_proj[:, 0].min())
max_x = max(w_left, corners_proj[:, 0].max())
min_y = min(0, corners_proj[:, 1].min())
max_y = max(h_left, corners_proj[:, 1].max())

pano_w = int(np.ceil(max_x - min_x))
pano_h = int(np.ceil(max_y - min_y))

# Translation matrix
T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

# Warp images
warped_left = cv2.warpPerspective(img_left, T, (pano_w, pano_h))
warped_right = cv2.warpPerspective(img_right, T @ H_LR, (pano_w, pano_h))

# Blend by averaging
panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
weights = np.zeros((pano_h, pano_w), dtype=np.float32)

for img in [warped_left, warped_right]:
    mask = (img.sum(axis=2) > 0).astype(np.float32)
    panorama += img.astype(np.float32) * mask[:, :, np.newaxis]
    weights += mask

weights[weights == 0] = 1
panorama = (panorama / weights[:, :, np.newaxis]).astype(np.uint8)

# Save result
cv2.imwrite('panaroma.jpg', panorama)
print(f"Panorama saved: {pano_w}x{pano_h}")
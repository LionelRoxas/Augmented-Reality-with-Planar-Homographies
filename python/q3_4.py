import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
import skimage.feature

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# Create the plot directly instead of using plotMatches
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
plt.axis('off')
skimage.feature.plot_matches(ax, cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY), 
                              cv2.cvtColor(cv_desk, cv2.COLOR_BGR2GRAY), 
                              locs1, locs2, matches, matches_color='r', only_matches=True)
plt.savefig('matched_features.jpg', dpi=150, bbox_inches='tight')
plt.show()
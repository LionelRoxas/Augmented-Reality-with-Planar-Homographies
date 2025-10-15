import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
from helper import plotMatches
import matplotlib.pyplot as plt
import skimage.feature

# Q3.5 - BRIEF and Rotations

# Read the image
cv_cover = cv2.imread('../data/cv_cover.jpg')

# Store match counts for each rotation angle
match_counts = []
angles = range(0, 360, 10)  # 0, 10, 20, ..., 350 degrees

# Angles we want to save visualizations for
save_angles = [100, 200, 300]

print("Testing BRIEF descriptor performance with rotations...")

for i in range(36):
    angle = i * 10
    print(f'{i+1}/36: Processing rotation {angle}°...')
    
    # Rotate Image (reshape=False to keep same dimensions)
    rot_img = rotate(cv_cover, angle, reshape=False)
    
    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, rot_img)
    
    # Update histogram - just count the number of matches
    num_matches = len(matches)
    match_counts.append(num_matches)
    print(f'  → {num_matches} matches found')
    
    # Save visualizations for specific angles
    if angle in save_angles:
        print(f'  Saving visualization for {angle}° rotation...')
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        plt.axis('off')
        
        # Convert to grayscale for visualization
        cv_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
        rot_gray = cv2.cvtColor(rot_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Plot matches
        skimage.feature.plot_matches(ax, cv_gray, rot_gray, 
                                      locs1, locs2, matches, 
                                      matches_color='r', only_matches=True)
        
        # Save the figure
        plt.savefig(f'rotate_{angle}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory

# Display histogram
plt.figure(figsize=(12, 6))
plt.bar(angles, match_counts, width=8, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Rotation Angle (degrees)', fontsize=12)
plt.ylabel('Number of Matches', fontsize=12)
plt.title('BRIEF Descriptor Performance vs Rotation Angle', fontsize=14)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(range(0, 360, 30))
plt.tight_layout()

# Save the histogram
plt.savefig('briefRotTest_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved images: rotate_100.png, rotate_200.png, rotate_300.png")
print(f"Saved histogram: briefRotTest_histogram.png")
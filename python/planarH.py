import numpy as np
import cv2

def computeH(x1, x2):
    # Q3.6
    # Compute the homography between two sets of points
    # x1: N x 2 matrix of (x,y) coordinates in image 1
    # x2: N x 2 matrix of (x,y) coordinates in image 2
    
    num_points = x1.shape[0]
    
    # Build matrix A for homography estimation
    A = []
    for i in range(num_points):
        x, y = x2[i]  # Point in image 2
        x_prime, y_prime = x1[i]  # Corresponding point in image 1
        
        # Each point pair gives 2 equations
        A.append([-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime])
        A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
    
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Solution is last column of V (last row of Vt)
    H2to1 = Vt[-1, :].reshape(3, 3)
    
    return H2to1


def computeH_norm(x1, x2):
    # Q3.7
    # Compute normalized homography between two sets of points
    
    # Compute the centroid of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)
    
    # Shift the origin of the points to the centroid
    x1_centered = x1 - mean1
    x2_centered = x2 - mean2
    
    # Normalize the points so that the largest distance from the origin is sqrt(2)
    max_dist1 = np.max(np.sqrt(np.sum(x1_centered**2, axis=1)))
    max_dist2 = np.max(np.sqrt(np.sum(x2_centered**2, axis=1)))
    
    scale1 = np.sqrt(2) / max_dist1
    scale2 = np.sqrt(2) / max_dist2
    
    x1_normalized = x1_centered * scale1
    x2_normalized = x2_centered * scale2
    
    # Similarity transform 1
    T1 = np.array([
        [scale1, 0, -scale1 * mean1[0]],
        [0, scale1, -scale1 * mean1[1]],
        [0, 0, 1]
    ])
    
    # Similarity transform 2
    T2 = np.array([
        [scale2, 0, -scale2 * mean2[0]],
        [0, scale2, -scale2 * mean2[1]],
        [0, 0, 1]
    ])
    
    # Compute homography on normalized points
    H_normalized = computeH(x1_normalized, x2_normalized)
    
    # Denormalization: H2to1 = T1^-1 * H_normalized * T2
    H2to1 = np.linalg.inv(T1) @ H_normalized @ T2
    
    return H2to1


def computeH_ransac(x1, x2, max_iters=500, inlier_tol=2.0):
    # Q3.8
    # Compute the best fitting homography given a list of matching points
    # x1: N x 2 matrix of points in image 1
    # x2: N x 2 matrix of corresponding points in image 2
    
    num_points = x1.shape[0]
    
    best_inlier_count = 0
    bestH2to1 = None
    inliers = np.zeros(num_points)
    
    for iteration in range(max_iters):
        # Randomly select 4 point correspondences
        indices = np.random.choice(num_points, 4, replace=False)
        x1_sample = x1[indices]
        x2_sample = x2[indices]
        
        # Compute homography from these 4 points
        H = computeH_norm(x1_sample, x2_sample)
        
        # Convert x2 to homogeneous coordinates
        x2_homogeneous = np.hstack([x2, np.ones((num_points, 1))])
        
        # Apply homography: x1_predicted = H @ x2
        x1_predicted_homogeneous = (H @ x2_homogeneous.T).T
        
        # Convert back to inhomogeneous coordinates
        x1_predicted = x1_predicted_homogeneous[:, :2] / x1_predicted_homogeneous[:, 2:3]
        
        # Compute error for all points
        errors = np.sqrt(np.sum((x1 - x1_predicted)**2, axis=1))
        
        # Count inliers (points with error below threshold)
        current_inliers = errors < inlier_tol
        inlier_count = np.sum(current_inliers)
        
        # Update best model if this one is better
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            bestH2to1 = H
            inliers = current_inliers.astype(float)
    
    # If no good homography found, return identity
    if bestH2to1 is None:
        bestH2to1 = np.eye(3)
        inliers = np.zeros(num_points)
    
    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography
    
    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1 * x_photo
    # For warping the template to the image, we need to invert it.
    
    # Create a copy of img to avoid modifying the original
    composite_img = img.copy()
    
    # Invert the homography
    H_inv = np.linalg.inv(H2to1)
    
    # Warp template using inverse homography
    warped_template = cv2.warpPerspective(template, H_inv, 
                                          (img.shape[1], img.shape[0]))
    
    # Create mask: non-black pixels in warped template
    mask = cv2.cvtColor(warped_template, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black out the region in original image where template will go
    img_bg = cv2.bitwise_and(composite_img, composite_img, mask=mask_inv)
    
    # Extract the template region
    template_fg = cv2.bitwise_and(warped_template, warped_template, mask=mask)
    
    # Combine background and foreground
    composite_img = cv2.add(img_bg, template_fg)
    
    return composite_img
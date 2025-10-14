import numpy as np
import cv2

def computeH(x1, x2):
	# Q3.6
	# Compute the homography between two sets of points
	num_points = x1.shape[0]
	x_invt, y_invt = x1[0]
	x, y = x2[0]
	A = np.array([[-x, -y, -1, 0, 0, 0, x_invt * x, x_invt * y, x_invt],
				  [0, 0, 0, -x, -y, -1, y_invt * x, y_invt * y, y_invt]])
	for i in range(num_points - 1):
		x_invt, y_invt = x1[i + 1]
		x, y = x2[i + 1]
		A_part = np.array([[-x, -y, -1, 0, 0, 0, x_invt * x, x_invt * y, x_invt],
						   [0, 0, 0, -x, -y, -1, y_invt * x, y_invt * y, y_invt]])
		A = np.concatenate((A, A_part), axis=0)

	U, S, V = np.linalg.svd(A.T @ A)
	H2to1 = np.reshape(V[8], (3, 3))

	return H2to1

def computeH_norm(x1, x2):
	# Q3.7
	# Compute the homography between two sets of points
	m1_x, m1_y = np.average(x1, axis=0)
	m2_x, m2_y = np.average(x2, axis=0)

	x1_a = x1 - np.average(x1, axis=0)
	x2_a = x2 - np.average(x2, axis=0)

	# Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	dist_max1 = np.max(np.sqrt(np.sum(np.square(x1_a), axis=1)))
	dist_max2 = np.max(np.sqrt(np.sum(np.square(x2_a), axis=1)))
	x1_a = np.sqrt(2) / dist_max1 * x1_a
	x2_a = np.sqrt(2) / dist_max2 * x2_a

	# Similarity transform
	T1 = np.sqrt(2) / dist_max1 * np.array([[1, 0, -m1_x], [0, 1, -m1_y], [0, 0, dist_max1 / np.sqrt(2)]])
	T2 = np.sqrt(2) / dist_max2 * np.array([[1, 0, -m2_x], [0, 1, -m2_y], [0, 0, dist_max2 / np.sqrt(2)]])

	# Compute homography
	H2to1 = computeH(x1_a, x2_a)

	# Denormalization
	H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H2to1, T2))
	return H2to1

def computeH_ransac(locs1, locs2):
	# Q3.8
	# Write a function: bestH2to1, inliers = computeH ransac(x1, x2) where bestH2to1 should be the homography H with most inliers found during RANSAC.
	num_points = locs1.shape[0]
	mat_one_all = np.ones((locs2.shape[0], 1))

	thr1 = 500
	thr2 = 0

	for i in range(1000):  # todo: check iteration number
		inliers_candid = np.ones(num_points)

		# Chooser 6 random points and compute the Homography matrix
		idx = np.random.randint(num_points, size=6)
		x1 = locs1[idx]
		x2 = locs2[idx]
		H2to1 = computeH_norm(x1, x2)

		locs2_new = np.concatenate((locs2, mat_one_all), axis=1)
		Hx2 = np.matmul(locs2_new, H2to1.T)
		Hx2_zremove = np.zeros_like(Hx2)
		Hx2_zremove[:, 0] = Hx2[:, 0] / Hx2[:, 2]
		Hx2_zremove[:, 1] = Hx2[:, 1] / Hx2[:, 2]
		Error_matrix = locs1 - Hx2[:, 0:2]
		Error = np.sqrt(np.sum(np.square(Error_matrix), axis=1))

		for j in range(num_points):
			if Error[j] > thr1:
				inliers_candid[j] = 0

		if inliers_candid.sum() > thr2:
			thr2 = inliers_candid.sum()
			inliers = inliers_candid
			bestH2to1 = H2to1

		if i == 999 and thr2 == 0:
			inliers = inliers_candid
			bestH2to1 = H2to1

	return bestH2to1, inliers

def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	template_alt = cv2.warpPerspective(template, H2to1, dsize=(img.shape[1], img.shape[0]))
	img2gray = cv2.cvtColor(template_alt, cv2.COLOR_BGR2GRAY)
	T, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	rows, cols, _ = template_alt.shape
	roi = img[0: 0 + rows, 0: 0 + cols]

	img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	template_fg = cv2.bitwise_and(template_alt, template_alt, mask=mask)

	# Use mask to combine the warped template and the image
	dst = cv2.add(img_bg, template_fg)
	img[0:0+rows, 0:0+cols] = dst

	return img
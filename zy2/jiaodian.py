import numpy as np
import cv2 as cv

filename = 'zy2\image.jpeg'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# Now draw them
res = np.hstack((centroids, corners))
res = np.intp(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]  # Mark centroids as red
img[res[:, 3], res[:, 2]] = [0, 255, 0]  # Mark refined corners as green

# Save the modified image
cv.imwrite('zy2\image_modified.jpeg', img)

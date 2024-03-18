import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import linear_model

def line_equation_from_points(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    magnitude = np.sqrt(delta_x**2 + delta_y**2)
    a = delta_x / magnitude
    b = -delta_y / magnitude
    d = (a * x1) + (b * y1)
    return a, b, d

# Load the image in grayscale
img = cv.imread("C:\\Users\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found"

# Apply edge detection using Canny
edges = cv.Canny(img, 550, 690)

# Get the coordinates of edge pixels
indicas = np.where(edges != 0)
x = indicas[1]
y = indicas[0]

# Reshape the data for sklearn compatibility
X = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of the estimated model
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

# Plotting the original image, edge image, and the RANSAC regressor
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
#plt.subplot(133), plt.imshow(img, cmap='gray')
plt.scatter(x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
plt.scatter(x[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=2, label="RANSAC Regressor")
plt.legend(loc="lower right")
plt.title("RANSAC Line Fitting")
plt.xticks([])
plt.yticks([])

plt.show()

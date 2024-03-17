import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the cropped image in grayscale
img = cv.imread("C:\\Users\\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)
#print(img.size)
assert img is not None, "Image not found"

# Apply Canny edge detection with adjusted parameters
edges = cv.Canny(img, 550, 690)

# Get the indices of non-zero (edge) pixels
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# Plot the original image and the edge image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([]), plt.yticks([])

plt.show()

# Implement Total Least Squares (TLS) method
data = np.vstack((x, y)).T
mean_x = np.mean(x)
mean_y = np.mean(y)
X = data - [mean_x, mean_y]
U, S, Vt = np.linalg.svd(X)
slope_tls = -Vt[0, 1] / Vt[0, 0]
intercept_tls = mean_y - slope_tls * mean_x

# Plot the scatter plot with the TLS line
plt.scatter(x, y)
plt.plot(x, slope_tls * x + intercept_tls, color='red', linewidth=1, label='Total Least Squares Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Total Least Squares Fit')
plt.legend()
plt.show()

print(f'Total Least Squares Fit line equation: y = {slope_tls}x + {intercept_tls}')

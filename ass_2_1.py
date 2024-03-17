import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the cropped image
img = cv2.imread("C:\\Users\\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection with adjusted parameters
edges = cv2.Canny(gray, 550, 690)

# Get the indices of non-zero (edge) pixels
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# Plot x and y in a scatter plot
plt.scatter(x, y, color='blue', label='Edge Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Edge Points Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()

# Fit a line to the edge points using least squares regression
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]  # Slope of the line
intercept = coefficients[1]  # Intercept of the line

# Calculate the least-squares-fit line
fit_line = slope * x + intercept

# Plot the scatter plot with the least-squares-fit line
plt.scatter(x, y, color='blue', label='Edge Points')
plt.plot(x, fit_line, color='red', label='Least-Squares-Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Edge Points with Least-Squares-Fit Line')
plt.legend()
plt.grid(True)
plt.show()

print(f'Estimated value of the crop field angle based on the least-squares-fit: {np.degrees(np.arctan(slope))}')

# Total least-squares-fit (Orthogonal Distance Regression)
# Concatenate x and y to form a matrix
data = np.vstack((x, y)).T

# Compute the total least-squares-fit line
_, _, V = np.linalg.svd(data - data.mean(axis=0))
total_slope = V[0, 1] / V[0, 0]
total_intercept = y.mean() - total_slope * x.mean()

# Calculate the total least-squares-fit line
total_fit_line = total_slope * x + total_intercept

# Plot the scatter plot with the total least-squares-fit line
plt.scatter(x, y, color='blue', label='Edge Points')
plt.plot(x, total_fit_line, color='green', label='Total Least-Squares-Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Edge Points with Total Least-Squares-Fit Line')
plt.legend()
plt.grid(True)
plt.show()

print(f'Estimated value of the crop field angle based on the total least-squares-fit: {np.degrees(np.arctan(total_slope))}')

# Proposed algorithm (Median Absolute Deviation)
# Compute median absolute deviation (MAD) of y values
median_y = np.median(y)
MAD = np.median(np.abs(y - median_y))

# Calculate the proposed line using MAD
proposed_slope = np.median((y - median_y) / (x - np.median(x)))
proposed_intercept = median_y - proposed_slope * np.median(x)
proposed_fit_line = proposed_slope * x + proposed_intercept

# Plot the scatter plot with the proposed line
plt.scatter(x, y, color='blue', label='Edge Points')
plt.plot(x, proposed_fit_line, color='purple', label='Proposed Line (MAD)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Edge Points with Proposed Line (MAD)')
plt.legend()
plt.grid(True)
plt.show()

print(f'Estimated value of the crop field angle based on the proposed algorithm (MAD): {np.degrees(np.arctan(proposed_slope))}')

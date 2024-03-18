import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:\\Users\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)

assert img is not None, "Image not found"
edges = cv.Canny(img, 550, 690)

indicas = np.where(edges != 0)
x = indicas[1]
y = indicas[0]

# Calculate least squares fit line
A = np.vstack([y, np.ones(len(y))]).T
m, b = np.linalg.lstsq(A, x, rcond=None)[0]

print("Least Squares Fit Line:")
print("Slope (m):", m)
print("Intercept (b):", b)

plt.scatter(y, x, c='b', marker='o', s=10, label='Edge Points')  # Change marker color to blue, marker style to circle, and marker size to 10
plt.plot(y, m*y + b, color='red', linewidth=1, label='Least Squares Fit')  # Plot the least squares fit line in red
plt.xlabel('X')  # Label for Y axis
plt.ylabel('Y')  # Label for X axis
plt.title('Scatter Plot with Least Squares Fit')  # Title for the plot
plt.legend()  # Show the legend
plt.show()

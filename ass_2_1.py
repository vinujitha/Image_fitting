import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

# Load the image in grayscale
img = cv.imread("C:\\Users\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)

assert img is not None, "Image not found"

# Apply edge detection using Canny
edges = cv.Canny(img, 550, 690)

# Get the coordinates of edge pixels
indicas = np.where(edges != 0)
x = indicas[1]
y = indicas[0]

# Create a DataFrame to tabulate x and y values
data = {'X': x, 'Y': y}
df = pd.DataFrame(data)

# Print the first few rows of the DataFrame (optional)
print(df.head())

# Plotting the original image and the edge image
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

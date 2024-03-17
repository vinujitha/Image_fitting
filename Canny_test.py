import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:\\Users\ASUS\\Downloads\\Crop_field_cropped.jpg\\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)
#plt.imshow(img, cmap='gray')
#plt.show() 
assert img is not None, "Image not found"
edges = cv.Canny(img,550,690)
#plt.imshow(edges, cmap='gray')
#plt.show()
#indicas = np.where(edges!=0)
#x = indicas[1]
#y = indicas[0]
#print(x)
#print(y)
#plt.scatter(x, y)
#plt.show()
#print(indicas)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.scatter(x, y) 
plt.show()


